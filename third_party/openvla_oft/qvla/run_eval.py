"""Evaluate QVLA mixed weights with official OFT two-camera/proprio continuous-action inference."""

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvla.artifacts import file_hash, now, protected_hashes, source_hashes, write_json
from qvla.evaluation_logs import parse_evaluation
from qvla.runtime import (
    bootstrap,
    configure_cuda,
    make_config,
    source_directory,
    stage_checkpoint,
    validate_model,
    working_directory,
)


def evaluate(args):
    import torch

    from qvla.quantization import apply_gates, load_gates

    gates_path = args.gates_path.expanduser().resolve() if args.gates_path else None
    assignment, metadata = load_gates(gates_path) if gates_path else ({}, {})
    channel_count = sum(map(len, assignment.values()))
    average_bits = sum(map(sum, assignment.values())) / channel_count if channel_count else 16.0
    output = args.local_log_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_cuda(args.memory_fraction)
    bootstrap(args.libero_root)
    import experiments.robot.libero.run_libero_eval as official

    checkpoint = stage_checkpoint(args.pretrained_checkpoint, output / "runtime_checkpoint")
    cfg = make_config(checkpoint, args.task_suite_name, args.seed, args.num_trials_per_task, output / "logs")
    cfg.run_id_note = args.run_id_note or ("qvla_a16" if gates_path else "bf16_reference")
    cfg.num_steps_wait = args.num_steps_wait
    config_path = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))) / "config.yaml"
    config_hash = file_hash(config_path) if config_path.is_file() else None
    state = {
        "status": "loading",
        "started": now(),
        "suite": args.task_suite_name,
        "seed": args.seed,
        "episodes_expected": None,
        "trials_per_task": args.num_trials_per_task,
        "activation_bits": 16,
        "activation_quantized_modules": 0,
        "channel_mean_bits": average_bits,
        "channel_count": channel_count,
        "weight_candidates": sorted({bit for values in assignment.values() for bit in values}),
        "gates_sha256": file_hash(gates_path) if gates_path else None,
        "allocation_stats": metadata.get("stats"),
        "memory_fraction_cap": args.memory_fraction,
        "OFT_images": 2,
        "OFT_proprio": True,
        "open_loop_steps": 8,
        "center_crop": True,
        "weight_dtype": "bfloat16",
        "full_precision": gates_path is None,
        "model_forward": "official bidirectional SDPA",
        "source_hashes": source_hashes(),
        "scope": "Local single-seed mixed fake-weight accuracy validation with BF16 compute; "
        "no native integer acceleration or exact-author calibration claim.",
    }
    write_json(output / "run.json", state)
    original_initialize = official.initialize_model
    original_log = official.log_message

    def initialize(config):
        with source_directory():
            model, head, proprio, noisy, processor = original_initialize(config)
        validate_model(model)
        before = protected_hashes(model, head, proprio)
        count = apply_gates(model, gates_path) if gates_path else 0
        if protected_hashes(model, head, proprio) != before:
            raise RuntimeError("Quantization changed a protected BF16 component")
        state.update(
            status="running",
            quantized_modules=count,
            actual_attention_classes=["LlamaSdpaAttention"],
            protected_parameters_unchanged=True,
            protected_parameters_sha256=before,
        )
        write_json(output / "run.json", state)
        return model, head, proprio, noisy, processor

    def log(message, log_file=None):
        original_log(message, log_file)
        if message.startswith("Episode error:"):
            raise RuntimeError(message)

    official.initialize_model = initialize
    official.log_message = log
    begin = time.monotonic()
    try:
        # Isolate video output per run. The initializer alone needs the source cwd.
        with working_directory(output):
            sr = float(official.eval_libero.__wrapped__(cfg))
        suite = official.benchmark.get_benchmark_dict()[args.task_suite_name]()
        tasks = [suite.get_task(index).language for index in range(suite.n_tasks)]
        (log_path,) = (output / "logs").glob("EVAL-*.txt")
        rows = parse_evaluation(log_path.read_text(), tasks, args.num_trials_per_task, args.task_suite_name)
        wins = sum(row["success"] for row in rows)
        if abs(sr - wins / len(rows)) > 1e-12:
            raise RuntimeError("Official return value and episode log disagree")
        if config_hash is not None and file_hash(config_path) != config_hash:
            raise RuntimeError("LIBERO configuration changed during evaluation")
        if (gates_path and file_hash(gates_path) != state["gates_sha256"]) or source_hashes() != state["source_hashes"]:
            raise RuntimeError("Gates or source code changed during evaluation")
        for row in rows:
            video = output / row["video"]
            if not video.is_file() or video.stat().st_size <= 1024:
                raise RuntimeError(f"Missing rollout video: {video}")
        write_json(output / "verified_episodes.json", rows)
        state.update(
            status="complete",
            finished=now(),
            episodes_expected=len(rows),
            episodes=len(rows),
            successes=wins,
            sr=sr,
            elapsed_seconds=time.monotonic() - begin,
            tasks=tasks,
            official_config=asdict(cfg),
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            LIBERO_config_unchanged=True,
            canonical_log=str(log_path.relative_to(output)),
            canonical_log_sha256=file_hash(log_path),
            verified_episodes_sha256=file_hash(output / "verified_episodes.json"),
        )
        write_json(output / "run.json", state)
        print(f"Verified {wins}/{len(rows)} episodes, SR={sr:.4%}", flush=True)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=now())
        write_json(output / "run.json", state)
        raise
    finally:
        official.initialize_model = original_initialize
        official.log_message = original_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", type=Path, required=True)
    precision = parser.add_mutually_exclusive_group(required=True)
    precision.add_argument("--gates_path", type=Path)
    precision.add_argument("--full_precision", action="store_true", help="Official BF16 policy, no quantization")
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--local_log_dir", type=Path, required=True)
    parser.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--run_id_note")
    parser.add_argument("--memory_fraction", type=float, default=0.42)
    args = parser.parse_args()
    if not 1 <= args.num_trials_per_task <= 50:
        parser.error("Use between 1 and 50 default initial states per task")
    if args.num_steps_wait < 0:
        parser.error("num_steps_wait must be nonnegative")
    evaluate(args)


if __name__ == "__main__":
    main()
