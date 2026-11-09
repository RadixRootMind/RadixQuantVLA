"""Evaluate validated OpenVLA fake-weight profiles with explicit causal attention."""

import argparse
import os
import time
from pathlib import Path

from qvla.artifacts import file_hash, now, protected_hashes, source_hashes, write_json
from qvla.evaluation_logs import parse_evaluation
from qvla.quantization import apply_gates, load_gates
from qvla.runtime import configure_cuda, load_model, load_processor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained_checkpoint", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gates_path", type=Path)
    mode.add_argument("--full_precision", action="store_true")
    p.add_argument("--local_log_dir", type=Path, required=True)
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--run_id_note", default="openvla_qvla")
    p.add_argument("--memory_fraction", type=float, default=0.42)
    p.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT"))
    p.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--flash_attn_path")
    args = p.parse_args()
    if not 1 <= args.num_trials_per_task <= 50 or args.num_steps_wait < 0:
        p.error("Invalid trial/wait count")
    import torch

    configure_cuda(args.memory_fraction)
    output = args.local_log_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = load_model(str(args.pretrained_checkpoint.resolve()), args.attention, args.libero_root, args.flash_attn_path)
    processor = load_processor(str(args.pretrained_checkpoint.resolve()))
    protected = protected_hashes(model)
    assigned, metadata = load_gates(args.gates_path) if args.gates_path else ({}, {})
    modules = apply_gates(model, args.gates_path) if args.gates_path else 0
    if protected_hashes(model) != protected:
        raise RuntimeError("Protected projector/language head changed")
    from experiments.robot.libero import run_libero_eval as official

    original_model, original_processor = official.get_model, official.get_processor
    original_action = official.get_action
    original_step_factory = official.get_libero_env
    config_path = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))) / "config.yaml"
    config_hash = file_hash(config_path)

    def action(*positional, **keywords):
        try:
            return original_action(*positional, **keywords)
        except Exception as exc:
            # Upstream catches Exception inside an episode. Escape that boundary
            # so a runtime fault is never reported as an unsuccessful policy trial.
            raise SystemExit(f"OpenVLA inference failed: {exc}") from exc

    def environment(*positional, **keywords):
        instance, description = original_step_factory(*positional, **keywords)

        class FailFastEnvironment:
            def __getattr__(self, name):
                return getattr(instance, name)

            def step(self, value):
                try:
                    return instance.step(value)
                except Exception as exc:
                    raise SystemExit(f"OpenVLA simulator failed: {exc}") from exc

        return FailFastEnvironment(), description

    official.get_model = lambda cfg: model
    official.get_processor = lambda cfg: processor
    official.get_action = action
    official.get_libero_env = environment
    cfg = official.GenerateConfig(
        pretrained_checkpoint=str(args.pretrained_checkpoint.resolve()),
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        center_crop=True,
        seed=args.seed,
        num_steps_wait=args.num_steps_wait,
        local_log_dir=str(output / "logs"),
        run_id_note=args.run_id_note,
        use_wandb=False,
    )
    channels = sum(map(len, assigned.values()))
    state = {
        "status": "running",
        "started": now(),
        "suite": args.task_suite_name,
        "seed": args.seed,
        "trials_per_task": args.num_trials_per_task,
        "channel_mean_bits": sum(map(sum, assigned.values())) / channels if channels else 16.0,
        "quantized_modules": modules,
        "activation_bits": 16,
        "activation_quantized_modules": 0,
        "weight_dtype": "bfloat16",
        "actual_attention_classes": model._qvla_attention_backend["actual_classes"],
        "protected_parameters_unchanged": True,
        "protected_parameters_sha256": protected,
        "gates_sha256": file_hash(args.gates_path) if args.gates_path else None,
        "source_hashes": source_hashes(),
        "num_images_in_input": 1,
        "use_proprio": False,
        "allocation_stats": metadata.get("stats"),
        "full_precision": args.full_precision,
        "scope": "Local weight-only fake quantization with explicit causal attention; no integer acceleration claim",
    }
    write_json(output / "run.json", state)
    previous = Path.cwd()
    begin = time.monotonic()
    try:
        os.chdir(output)
        official.eval_libero.__wrapped__(cfg)
        suite = official.benchmark.get_benchmark_dict()[args.task_suite_name]()
        tasks = [suite.get_task(index).language for index in range(suite.n_tasks)]
        (log,) = (output / "logs").glob("EVAL-*.txt")
        rows = parse_evaluation(log.read_text(), tasks, args.num_trials_per_task, args.task_suite_name)
        if protected_hashes(model) != protected or source_hashes() != state["source_hashes"]:
            raise RuntimeError("Protected weights or source changed during evaluation")
        if file_hash(config_path) != config_hash:
            raise RuntimeError("LIBERO config changed during evaluation")
        for row in rows:
            video = output / row["video"]
            if not video.is_file() or video.stat().st_size <= 1024:
                raise RuntimeError("Missing rollout video")
        write_json(output / "verified_episodes.json", rows)
        wins = sum(row["success"] for row in rows)
        state.update(
            status="complete",
            finished=now(),
            episodes=len(rows),
            successes=wins,
            sr=wins / len(rows),
            tasks=tasks,
            canonical_log=str(log.relative_to(output)),
            canonical_log_sha256=file_hash(log),
            verified_episodes_sha256=file_hash(output / "verified_episodes.json"),
            LIBERO_config_unchanged=True,
            elapsed_seconds=time.monotonic() - begin,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
        )
        write_json(output / "run.json", state)
    except BaseException as exc:
        state.update(status="failed", finished=now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "run.json", state)
        raise
    finally:
        os.chdir(previous)
        official.get_model, official.get_processor = original_model, original_processor
        official.get_action = original_action
        official.get_libero_env = original_step_factory


if __name__ == "__main__":
    main()
