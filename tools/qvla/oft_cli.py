"""Radix stage adapter for the OpenVLA and OpenVLA-OFT QVLA routes.

The historical filename and .oft_* stage manifests are retained for compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def is_oft_backend(root):
    return bool(root) and (Path(root) / "prismatic/models/action_heads.py").is_file()


def maybe_dispatch(stage):
    """Dispatch legacy CLI entrypoints to the selected backend's QVLA package."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--backend-root",
        default=os.environ.get("QVLA_BACKEND_ROOT") or ROOT / "third_party/openvla",
    )
    known, _ = parser.parse_known_args()
    if not (Path(known.backend_root) / "qvla/calibrate_actions.py").is_file():
        return False
    main(["--stage", stage, *sys.argv[1:]])
    return True


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@cache
def checkpoint_identity(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("A local merged OpenVLA-family checkpoint is required")
    suffixes = {".json", ".py", ".model", ".safetensors", ".pt", ".bin"}
    return {
        path.name: sha(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in suffixes
    }


def source_identity(backend):
    paths = [
        path
        for folder in ("qvla", "prismatic", "experiments")
        for path in (backend / folder).rglob("*.py")
    ]
    paths += list((ROOT / "tools/qvla").glob("*.py"))
    paths += [
        Path(__file__),
        ROOT / "scripts/run_openvla_qvla.sh",
        ROOT / "scripts/run_awesome_quant_vla.sh",
        ROOT / "scripts/common_paths.sh",
        backend / "pyproject.toml",
        backend / "qvla/requirements.txt",
    ]
    result = {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}
    spec = importlib.util.find_spec("transformers")
    if spec and spec.origin:
        result["installed_transformers/modeling_llama.py"] = sha(
            Path(spec.origin).parent / "models/llama/modeling_llama.py"
        )
    return result


def stage(args, name, inputs, artifact, command):
    """Reuse only a successful artifact with the exact same recorded inputs."""
    metadata = args.output_root / f".oft_{name}.json"
    fingerprint = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode()
    ).hexdigest()
    if artifact.exists():
        if metadata.is_file():
            previous = json.loads(metadata.read_text())
            if (
                previous.get("status") == "complete"
                and previous.get("fingerprint") == fingerprint
                and previous.get("artifact_sha256") == sha(artifact)
            ):
                print(f"[qvla] Reusing verified {name}: {artifact}", flush=True)
                return
        raise ValueError(
            f"Unverified or stale {name} artifact: {artifact}; use a new output directory"
        )
    if metadata.exists():
        raise ValueError(
            f"Previous {name} attempt exists; preserve it and use a new output directory"
        )
    state = {
        "status": "running",
        "fingerprint": fingerprint,
        "inputs": inputs,
        "command": list(map(str, command)),
        "artifact": str(artifact),
    }
    write(metadata, state)
    try:
        env = {
            **os.environ,
            "PYTHONPATH": str(args.backend_root)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        subprocess.run(
            list(map(str, command)), cwd=args.backend_root, env=env, check=True
        )
        if not artifact.is_file():
            raise RuntimeError(f"Missing {name} artifact: {artifact}")
        state.update(status="complete", artifact_sha256=sha(artifact))
        write(metadata, state)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write(metadata, state)
        raise


def module(name, *args):
    return [sys.executable, "-u", "-m", "qvla." + name, *map(str, args)]


def run(args):
    args.backend_root = args.backend_root.expanduser().resolve()
    oft = is_oft_backend(args.backend_root)
    if not (args.backend_root / "qvla/calibrate_actions.py").is_file():
        raise ValueError("Select a supported OpenVLA/OpenVLA-OFT backend")
    route = "openvla_oft" if oft else "openvla"
    args.attention = args.attention or (
        "sdpa"
        if oft or args.method == "fp16" or args.target_avg_bits < 8
        else "flash_attention_2"
    )
    args.calibration_attention = args.calibration_attention or "sdpa"
    if args.restore_pruned_fraction is None:
        args.restore_pruned_fraction = (
            0.25 if not oft and args.target_avg_bits == 8 else 0.0
        )
    if oft and (
        args.attention != "sdpa"
        or args.calibration_attention != "sdpa"
        or args.restore_pruned_fraction
    ):
        raise ValueError(
            "OFT uses bidirectional SDPA and no OpenVLA-specific pruning repair"
        )
    if args.method == "qvla" and set(map(int, args.bits.split(","))) - {0, 4, 8, 16}:
        raise ValueError(
            "This configuration excludes 2bit; choose weights from 0,4,8,16"
        )
    if args.proxy_bits and set(map(int, args.proxy_bits.split(","))) != {0, 4, 8, 16}:
        raise ValueError("Action calibration requires proxy bits 0,4,8,16")
    if args.max_layers is not None or args.resume or args.percdamp is not None:
        raise ValueError(
            "Input-Hessian max_layers/resume/percdamp options are not action-calibration options"
        )
    if args.center_crop.lower() != "true" or args.model_family != "openvla":
        raise ValueError(
            "This route uses the official LIBERO center-crop/openvla configuration"
        )
    if args.device not in ("cuda", "cuda:0") or args.dtype not in ("bf16", "bfloat16"):
        raise ValueError(
            "Select one visible CUDA GPU; inference and fake weights use BF16"
        )
    if args.output_root is None:
        location = (
            args.calib_jsonl
            if args.stage == "build_calib"
            else (
                args.proxy_pt
                if args.stage == "proxy"
                else args.gates_json
                if args.stage == "assign"
                else args.export_dir
                if args.stage == "export"
                else args.log_dir
            )
        )
        if location is None:
            raise ValueError(
                "Pass --output-root, or an explicit artifact path for the selected stage"
            )
        parent = location.expanduser().resolve().parent
        args.output_root = (
            parent.parent
            if parent.name in ("calib", "proxy", "gates", "eval")
            else parent
        )
    args.output_root = args.output_root.expanduser().resolve()
    args.calib_jsonl = (
        (args.calib_jsonl or args.output_root / "calib/records.jsonl")
        .expanduser()
        .resolve()
    )
    args.proxy_pt = (
        (args.proxy_pt or args.output_root / "proxy/proxy.pt").expanduser().resolve()
    )
    args.gates_json = (
        (args.gates_json or args.output_root / "gates/greedy_bits.json")
        .expanduser()
        .resolve()
    )
    args.log_dir = (
        (args.log_dir or args.output_root / "logs/eval").expanduser().resolve()
    )
    args.export_dir = (
        (args.export_dir or args.output_root / "export/fakew").expanduser().resolve()
    )
    if args.stage == "plan":
        print(
            json.dumps(
                {
                    "backend": route,
                    "method": args.method,
                    "samples": args.samples,
                    "bits": args.bits,
                    "target_avg_bits": args.target_avg_bits,
                    "max_prune_ratio": args.max_prune_ratio,
                    "attention": args.attention,
                    "calibration_attention": args.calibration_attention,
                    "restore_pruned_fraction": args.restore_pruned_fraction,
                    "trials_per_task": args.trials,
                    "data_root": str(args.data_root),
                    "output_root": str(args.output_root),
                },
                indent=2,
            )
        )
        return
    if args.stage == "result":
        print((args.output_root / "merged_summary.json").read_text())
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = source_identity(args.backend_root)
    versions = {
        package: importlib.metadata.version(package)
        for package in (
            "torch",
            "transformers",
            "numpy",
            "timm",
            "tensorflow-datasets",
            "mujoco",
            "robosuite",
        )
    }
    common = {
        "sources": sources,
        "environment": versions,
        "seed": args.seed,
        "flash_attn_path": str(Path(args.flash_attn_path).expanduser().resolve())
        if args.flash_attn_path and not oft
        else None,
    }
    libero = ["--libero_root", args.libero_root] if args.libero_root else []
    calibration_options = [] if oft else ["--attention", args.calibration_attention]
    evaluation_options = [] if oft else ["--attention", args.attention]
    if args.flash_attn_path and not oft:
        calibration_options += ["--flash_attn_path", args.flash_attn_path]
        evaluation_options += ["--flash_attn_path", args.flash_attn_path]

    def build():
        if args.data_root is None:
            raise ValueError(
                "Calibration needs --rlds-root/--data-root or QVLA_RLDS_ROOT; simulator reset images are insufficient"
            )
        arguments = [
            "--data_root",
            args.data_root.resolve(),
            "--out_dir",
            args.calib_jsonl.parent,
            "--manifest_name",
            args.calib_jsonl.name,
            "--trajectories",
            args.samples,
            "--seed",
            args.seed,
        ]
        if args.indices_jsonl:
            arguments += ["--indices_jsonl", args.indices_jsonl.resolve()]
        infos = list(
            args.data_root.resolve().glob("libero_*_no_noops/1.0.0/dataset_info.json")
        )
        if len(infos) < 4:
            raise ValueError(
                "Expected downloaded Spatial/Object/Goal/10 modified LIBERO RLDS datasets"
            )
        inputs = {
            **common,
            "samples": args.samples,
            "data_root": str(args.data_root.resolve()),
            "dataset_info": {str(path): sha(path) for path in infos},
            "indices_sha256": sha(args.indices_jsonl) if args.indices_jsonl else None,
        }
        stage(
            args,
            "build_calib",
            inputs,
            args.calib_jsonl,
            module("prepare_calibration", *arguments),
        )
        rows = [
            json.loads(line)
            for line in args.calib_jsonl.read_text().splitlines()
            if line.strip()
        ]
        if len(rows) != args.samples:
            raise ValueError(
                f"Prepared {len(rows)}/{args.samples} requested training frames"
            )

    def calibrate():
        inputs = {
            **common,
            "checkpoint": checkpoint_identity(str(args.checkpoint.resolve())),
            "records_sha256": sha(args.calib_jsonl),
            "samples": args.samples,
            "attention": args.calibration_attention,
        }
        stage(
            args,
            "proxy",
            inputs,
            args.proxy_pt,
            module(
                "calibrate_actions",
                "--pretrained_checkpoint",
                args.checkpoint.resolve(),
                "--calib_jsonl",
                args.calib_jsonl,
                "--out_dir",
                args.proxy_pt.parent,
                "--max_samples",
                args.samples,
                "--seed",
                args.seed,
                "--task_suite_name",
                args.suite,
                "--memory_fraction",
                args.calibration_memory_fraction,
                *calibration_options,
                *libero,
            ),
        )
        status = json.loads((args.proxy_pt.parent / "status.json").read_text())
        if (
            status["status"] != "complete"
            or status["completed_samples"] != args.samples
        ):
            raise ValueError("Calibration did not complete all requested records")

    def allocate():
        inputs = {
            **common,
            "proxy_sha256": sha(args.proxy_pt),
            "bits": args.bits,
            "target_avg_bits": args.target_avg_bits,
            "max_prune_ratio": args.max_prune_ratio,
            "restore_pruned_fraction": args.restore_pruned_fraction,
        }
        stage(
            args,
            "assign",
            inputs,
            args.gates_json,
            module(
                "assign_gates_from_sensitivity",
                "--proxy_pt",
                args.proxy_pt,
                "--bits",
                args.bits,
                "--target_avg_bits",
                args.target_avg_bits,
                "--max_prune_ratio",
                args.max_prune_ratio,
                "--out_json",
                args.gates_json,
                *(
                    []
                    if oft
                    else ["--restore_pruned_fraction", args.restore_pruned_fraction]
                ),
            ),
        )

    def evaluate():
        inputs = {
            **common,
            "checkpoint": checkpoint_identity(str(args.checkpoint.resolve())),
            "gates_sha256": sha(args.gates_json) if args.method == "qvla" else None,
            "suite": args.suite,
            "trials": args.trials,
            "num_steps_wait": args.num_steps_wait,
            "method": args.method,
            "attention": args.attention,
            "inference_memory_fraction": args.inference_memory_fraction,
        }
        precision = (
            ["--gates_path", args.gates_json]
            if args.method == "qvla"
            else ["--full_precision"]
        )
        stage(
            args,
            "eval",
            inputs,
            args.log_dir / "run.json",
            module(
                "run_eval",
                "--pretrained_checkpoint",
                args.checkpoint.resolve(),
                *precision,
                "--task_suite_name",
                args.suite,
                "--num_trials_per_task",
                args.trials,
                "--local_log_dir",
                args.log_dir,
                "--seed",
                args.seed,
                "--num_steps_wait",
                args.num_steps_wait,
                "--memory_fraction",
                args.inference_memory_fraction,
                "--run_id_note",
                args.run_id_note,
                *evaluation_options,
                *libero,
            ),
        )
        run_record = json.loads((args.log_dir / "run.json").read_text())
        if run_record["status"] != "complete":
            raise ValueError("Evaluator did not complete its episode/log verification")
        env = {
            **os.environ,
            "PYTHONPATH": str(args.backend_root)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        subprocess.run(
            module("verify_eval", "--run_dir", args.log_dir),
            cwd=args.backend_root,
            env=env,
            check=True,
        )
        if (
            run_record["trials_per_task"] != args.trials
            or run_record["seed"] != args.seed
        ):
            raise ValueError("Evaluator protocol mismatch")
        summary = {
            "task_suite": args.suite,
            "benchmark": args.profile or route,
            "quantization": f"QVLA-W{run_record['channel_mean_bits']:g}"
            if args.method == "qvla"
            else "BF16",
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "total_episodes": run_record["episodes"],
            "total_successes": run_record["successes"],
            "total_success_rate": run_record["sr"],
            "seed": args.seed,
            "source_log": str(args.log_dir / run_record["canonical_log"]),
            "verified_run": str(args.log_dir / "run.json"),
            "verified_run_sha256": sha(args.log_dir / "run.json"),
            "actual_weight_dtype": "bfloat16",
            "activation_bits": 16,
        }
        write(args.output_root / "merged_summary.json", summary)
        (args.output_root / "merged_summary.md").write_text(
            f"# LIBERO {args.suite}\n\n{summary['total_successes']}/{summary['total_episodes']} successes "
            f"({100 * summary['total_success_rate']:.1f}%).\n\n"
            f"Quantization: {summary['quantization']}; seed={args.seed}; verified OpenVLA-family evaluation.\n"
        )

    if args.stage == "run":
        if args.method == "qvla":
            build()
            calibrate()
            allocate()
        evaluate()
    elif args.stage == "build_calib":
        build()
    elif args.stage == "proxy":
        calibrate()
    elif args.stage == "assign":
        allocate()
    elif args.stage == "eval":
        evaluate()
    elif args.stage == "export":
        stage(
            args,
            "export",
            {
                **common,
                "checkpoint": checkpoint_identity(str(args.checkpoint.resolve())),
                "gates_sha256": sha(args.gates_json),
            },
            args.export_dir / "qvla_export.json",
            module(
                "inject_fake_w",
                "--pretrained_checkpoint",
                args.checkpoint.resolve(),
                "--gates_path",
                args.gates_json,
                "--out_dir",
                args.export_dir,
                *evaluation_options,
                *libero,
            ),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "plan",
            "result",
            "build_calib",
            "proxy",
            "assign",
            "eval",
            "export",
            "run",
        ],
        required=True,
    )
    parser.add_argument(
        "--backend-root",
        type=Path,
        default=os.environ.get("QVLA_BACKEND_ROOT") or ROOT / "third_party/openvla",
    )
    parser.add_argument("--method", choices=["qvla", "fp16"], default="qvla")
    parser.add_argument("--profile", default=os.environ.get("PROFILE"))
    parser.add_argument(
        "--attention",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=os.environ.get("QVLA_ATTENTION") or None,
    )
    parser.add_argument(
        "--calibration-attention",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=os.environ.get("QVLA_CALIBRATION_ATTENTION") or None,
    )
    parser.add_argument(
        "--restore-pruned-fraction",
        type=float,
        default=os.environ.get("QVLA_RESTORE_PRUNED_FRACTION") or None,
    )
    parser.add_argument(
        "--flash-attn-path", default=os.environ.get("OPENVLA_FLASH_ATTN_PATH") or None
    )
    parser.add_argument("--pretrained_checkpoint", type=Path, dest="checkpoint")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--calib_jsonl", "--out-jsonl", type=Path)
    parser.add_argument("--proxy_pt", "--out_path", type=Path)
    parser.add_argument("--gates_path", "--out_json", type=Path, dest="gates_json")
    parser.add_argument("--local_log_dir", type=Path, dest="log_dir")
    parser.add_argument("--out_dir", type=Path, dest="export_dir")
    parser.add_argument(
        "--data-root",
        "--rlds-root",
        type=Path,
        default=os.environ.get("QVLA_RLDS_ROOT") or None,
    )
    parser.add_argument(
        "--indices-jsonl",
        "--calib-indices",
        type=Path,
        dest="indices_jsonl",
        default=os.environ.get("QVLA_CALIB_INDICES") or None,
    )
    parser.add_argument(
        "--task_suite_name", "--task-suite-name", dest="suite", default="libero_spatial"
    )
    parser.add_argument(
        "--num_trials_per_task", "--trials", dest="trials", type=int, default=50
    )
    parser.add_argument(
        "--max_samples",
        "--max-samples",
        "--num-samples",
        dest="samples",
        type=int,
        default=512,
    )
    parser.add_argument("--bits", default="0,4,8,16")
    parser.add_argument("--proxy-bits")
    parser.add_argument(
        "--target_avg_bits", "--target-avg-bits", type=float, default=8.0
    )
    parser.add_argument(
        "--max-prune-ratio", "--max_prune_ratio", type=float, default=0.01
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--libero_root", "--libero-root", default=os.environ.get("LIBERO_ROOT")
    )
    parser.add_argument("--center-crop", default="True")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--run-id-note", default="radix_qvla")
    parser.add_argument("--model-family", default="openvla")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--calibration-memory-fraction", type=float, default=0.55)
    parser.add_argument("--inference-memory-fraction", type=float, default=0.42)
    parser.add_argument("--max_layers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--percdamp", type=float)
    args = parser.parse_args(argv)
    if args.stage in ("proxy", "eval", "export", "run") and args.checkpoint is None:
        parser.error("pretrained_checkpoint is required for this stage")
    if args.samples < 1 or not 1 <= args.trials <= 50 or args.num_steps_wait < 0:
        parser.error(
            "Require positive samples, trials in1..50, and nonnegative steps-wait"
        )
    run(args)


if __name__ == "__main__":
    main()
