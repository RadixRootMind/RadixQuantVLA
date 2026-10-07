#!/usr/bin/env python

import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


CONDA_ROOT = Path(os.environ.get("CONDA_ROOT", os.path.join(os.path.expanduser("~"), "miniconda3"))).resolve()
DEFAULT_CONDA_ENV = os.environ.get(
    "AWESOME_QVLA_CONDA_ENV",
    os.environ.get("QUANTVLA_CONDA_ENV", "awesome_quant_vla"),
)
GROOT_CONDA_ENV = os.environ.get("GROOT_CONDA_ENV", DEFAULT_CONDA_ENV)
LIBERO_CONDA_ENV = os.environ.get("LIBERO_CONDA_ENV", GROOT_CONDA_ENV)
GROOT_PY = os.environ.get("GROOT_PY", str(CONDA_ROOT / "envs" / GROOT_CONDA_ENV / "bin" / "python"))
LIBERO_PY = os.environ.get("LIBERO_PY", str(CONDA_ROOT / "envs" / LIBERO_CONDA_ENV / "bin" / "python"))
QUANTVLA_ROOT = Path(
    os.environ.get(
        "AWESOME_QVLA_ROOT",
        os.environ.get("QUANTVLA_ROOT", str(Path(__file__).resolve().parents[1])),
    )
).resolve()
LIBERO_ROOT = Path(os.environ.get("LIBERO_ROOT", os.path.join(os.path.expanduser("~"), "LIBERO"))).resolve()


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_json_list(name: str) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{name} must be a JSON array of strings")
    return parsed


def suite_defaults(task_suite: str) -> tuple[str, str, int]:
    if task_suite == "libero_spatial":
        return (
            "youliangtan/gr00t-n1.5-libero-spatial-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            10,
        )
    if task_suite == "libero_goal":
        return (
            "youliangtan/gr00t-n1.5-libero-goal-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfigMeanStd",
            10,
        )
    if task_suite == "libero_object":
        return (
            "youliangtan/gr00t-n1.5-libero-object-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            10,
        )
    if task_suite == "libero_90":
        return (
            "youliangtan/gr00t-n1.5-libero-90-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            90,
        )
    if task_suite == "libero_10":
        return (
            "youliangtan/gr00t-n1.5-libero-long-posttrain",
            "examples.Libero.custom_data_config:LiberoDataConfig",
            10,
        )
    raise ValueError(f"Unknown TASK_SUITE: {task_suite}")


def wait_for_port(port: int, timeout_s: int = 3600, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Server process exited with code {proc.returncode} before opening port {port}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"Timed out waiting for localhost:{port}")


def shard_tasks(num_tasks: int, num_shards: int) -> list[list[int]]:
    shards = [[] for _ in range(num_shards)]
    for idx, task_id in enumerate(range(num_tasks)):
        shards[idx % num_shards].append(task_id)
    return shards


def print_log_tail(path: Path, lines: int = 120) -> None:
    if not path.exists():
        return
    print(f"----- tail -n {lines} {path} -----", flush=True)
    try:
        content = path.read_text(errors="ignore").splitlines()
        for line in content[-lines:]:
            print(line, flush=True)
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}", flush=True)


def print_shard_log_tails(output_root: Path, shard_idx: int, task_suite: str) -> None:
    logs_dir = output_root / "logs"
    candidates = [
        logs_dir / f"driver_shard_{shard_idx}.log",
        logs_dir / f"server_shard_{shard_idx}.log",
        logs_dir / f"eval_shard_{shard_idx}.stdout.log",
    ]
    candidates.extend(sorted((logs_dir / "eval").glob(f"*{task_suite}*shard_{shard_idx}*.log")))

    seen: set[Path] = set()
    print(f"[Shard {shard_idx}] failed; showing recent shard logs", flush=True)
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        print_log_tail(path)


def make_base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("VIRTUAL_ENV", None)
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")

    pythonpath_entries = [str(QUANTVLA_ROOT)]
    if LIBERO_ROOT.exists():
        pythonpath_entries.append(str(LIBERO_ROOT))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    cache_root = env.get(
        "AWESOME_QVLA_CACHE_ROOT",
        env.get("QUANTVLA_CACHE_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "awesome_quant_vla")),
    )
    env["AWESOME_QVLA_CACHE_ROOT"] = cache_root
    env["QUANTVLA_CACHE_ROOT"] = cache_root
    env["HF_HOME"] = f"{cache_root}/huggingface"
    env["HUGGINGFACE_HUB_CACHE"] = f"{cache_root}/huggingface/hub"
    env["TRANSFORMERS_CACHE"] = f"{cache_root}/huggingface/transformers"
    env["TORCH_HOME"] = f"{cache_root}/torch"
    env["XDG_CACHE_HOME"] = f"{cache_root}/xdg"
    env["LIBERO_CONFIG_PATH"] = env.get("LIBERO_CONFIG_PATH", os.path.join(os.path.expanduser("~"), ".libero"))
    env["LIBERO_ROOT"] = str(LIBERO_ROOT)
    env["MPLCONFIGDIR"] = f"{cache_root}/matplotlib"
    env["GR00T_CPU_THREADS"] = env.get("GR00T_CPU_THREADS", "4")
    env["GR00T_INTEROP_THREADS"] = env.get("GR00T_INTEROP_THREADS", "1")
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", env["GR00T_CPU_THREADS"])
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", env["GR00T_CPU_THREADS"])
    env["OPENBLAS_NUM_THREADS"] = env.get("OPENBLAS_NUM_THREADS", env["GR00T_CPU_THREADS"])
    env["NUMEXPR_NUM_THREADS"] = env.get("NUMEXPR_NUM_THREADS", env["GR00T_CPU_THREADS"])
    env["OMP_WAIT_POLICY"] = env.get("OMP_WAIT_POLICY", "PASSIVE")

    for key in (
        env["HF_HOME"],
        env["HUGGINGFACE_HUB_CACHE"],
        env["TRANSFORMERS_CACHE"],
        env["TORCH_HOME"],
        env["XDG_CACHE_HOME"],
        env["LIBERO_CONFIG_PATH"],
        env["MPLCONFIGDIR"],
    ):
        Path(key).mkdir(parents=True, exist_ok=True)
    return env


def configure_duquant_env(env: dict[str, str], wbits: int, abits: int, packdir: str) -> None:
    # Set torch-compile disable flags universally (needed regardless of method).
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_CUDA_GRAPH_DISABLE"] = "1"
    env["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
    # Skip DuQuant-specific env injection when method does not need it.
    # METHOD=fp16: pure FP16, no quant at all.
    # METHOD=gptq / pure_gptq: GPTQ pack codepath. When not hybrid, strip any
    # inherited DuQuant env so the model loads cleanly; when hybrid, the caller's
    # DuQuant env (e.g. LLM side) is kept and composes with the GPTQ pack.
    method = env.get("METHOD", "").lower()
    hybrid = env.get("GR00T_HYBRID_QUANT", "0") not in ("0", "false", "False")
    if method in ("fp16", "gptq", "pure_gptq", "rtn") and not hybrid:
        # Strip any inherited DuQuant env so model loads cleanly.
        for k in list(env.keys()):
            if k.startswith("GR00T_DUQUANT_") and k != "GR00T_DUQUANT_PACKDIR":
                env.pop(k, None)
        return
    if method in ("fp16", "gptq", "pure_gptq") and hybrid:
        # HYBRID mode: keep caller's DuQuant env vars; only inject ABITS from $abits
        # so LLM bit-width follows the user-requested precision.
        env["GR00T_DUQUANT_ABITS"] = str(abits)
        env["GR00T_DUQUANT_WBITS_DEFAULT"] = str(wbits)
        env["GR00T_DUQUANT_PACKDIR"] = packdir
        return
    env["GR00T_DUQUANT_DEBUG"] = env.get("GR00T_DUQUANT_DEBUG", "1")
    env["GR00T_DUQUANT_SCOPE"] = env.get("GR00T_DUQUANT_SCOPE", "")
    env["GR00T_DUQUANT_INCLUDE"] = env.get("GR00T_DUQUANT_INCLUDE", (
        r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        r"|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*"
    ))
    env["GR00T_DUQUANT_EXCLUDE"] = env.get(
        "GR00T_DUQUANT_EXCLUDE",
        r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)",
    )
    env["GR00T_DUQUANT_WBITS_DEFAULT"] = str(wbits)
    env["GR00T_DUQUANT_ABITS"] = str(abits)
    env["GR00T_DUQUANT_BLOCK"] = env.get("GR00T_DUQUANT_BLOCK", "64")
    env["GR00T_DUQUANT_BLOCK_OUT"] = env.get("GR00T_DUQUANT_BLOCK_OUT", env["GR00T_DUQUANT_BLOCK"])
    env["GR00T_DUQUANT_PERMUTE"] = env.get("GR00T_DUQUANT_PERMUTE", "0")
    env["GR00T_DUQUANT_ROW_ROT"] = env.get("GR00T_DUQUANT_ROW_ROT", "restore")
    env["GR00T_DUQUANT_ACT_PCT"] = env.get("GR00T_DUQUANT_ACT_PCT", "99.9")
    env["GR00T_DUQUANT_CALIB_STEPS"] = env.get("GR00T_DUQUANT_CALIB_STEPS", "32")
    env["GR00T_DUQUANT_LS"] = env.get("GR00T_DUQUANT_LS", "0.15")
    env["GR00T_DUQUANT_PACKDIR"] = packdir
    # Preserve explicitly requested Pi0.5 calibration through the launcher.
    # Keep the historical non-Pi0.5 benchmark default neutral.
    if env.get("GR00T_ATM_SCOPE") == "pi05":
        env.setdefault("GR00T_ATM_ENABLE", "0")
        env.setdefault("GR00T_OHB_ENABLE", "0")
    else:
        env["GR00T_ATM_ENABLE"] = "0"
        env["GR00T_OHB_ENABLE"] = "0"
        for key in ("GR00T_ATM_ALPHA_PATH", "GR00T_ATM_SCOPE", "GR00T_OHB_FALLBACK", "GR00T_OHB_SCOPE"):
            env.pop(key, None)
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHDYNAMO_DISABLE"] = "1"
    env["TORCH_CUDA_GRAPH_DISABLE"] = "1"
    env["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"


def run_shard(
    shard_idx: int,
    gpu: str,
    port: int,
    task_ids: list[int],
    task_suite: str,
    model_path: str,
    data_config: str,
    num_trials: int,
    num_steps_wait: int,
    denoising_steps: int,
    wbits: int,
    abits: int,
    output_root: Path,
    packdir: str,
    headless: bool,
    init_offset: int = 0,
) -> None:
    base_env = make_base_env()
    base_env["CUDA_VISIBLE_DEVICES"] = gpu
    configure_duquant_env(base_env, wbits=wbits, abits=abits, packdir=packdir)
    # Pass calibration suite name through to inference_service.py so
    # policy.warmup() can load diverse task prompts (not just "warmup" token).
    base_env["GR00T_CALIB_SUITE"] = task_suite
    # Per-shard activation stats dump dir (post-A8-calibration percentile vectors).
    act_stats_subdir = output_root / "act_stats" / f"shard_{shard_idx}"
    act_stats_subdir.mkdir(parents=True, exist_ok=True)
    base_env["GR00T_ACT_STATS_DIR"] = str(act_stats_subdir)
    inference_py = env_str("INFERENCE_PY", GROOT_PY)
    inference_script = env_str(
        "INFERENCE_SCRIPT", str(QUANTVLA_ROOT / "scripts" / "inference_service.py")
    )
    inference_cwd = env_str("INFERENCE_CWD", str(QUANTVLA_ROOT))
    inference_extra_args = env_json_list("INFERENCE_EXTRA_ARGS_JSON")
    eval_py = env_str("EVAL_PY", LIBERO_PY)
    eval_script = env_str(
        "EVAL_SCRIPT", str(QUANTVLA_ROOT / "examples" / "Libero" / "eval" / "run_libero_eval.py")
    )
    eval_cwd = env_str("EVAL_CWD", str(Path(eval_script).resolve().parent))
    eval_extra_args = env_json_list("EVAL_EXTRA_ARGS_JSON")

    logs_dir = output_root / "logs"
    eval_log_dir = logs_dir / "eval"
    summaries_dir = output_root / "summaries"
    rollout_dir = output_root / "rollouts" / f"shard_{shard_idx}"
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    server_log_path = logs_dir / f"server_shard_{shard_idx}.log"
    eval_stdout_log_path = logs_dir / f"eval_shard_{shard_idx}.stdout.log"
    summary_json_path = summaries_dir / f"shard_{shard_idx}.json"

    with open(server_log_path, "w") as server_log:
        server_proc = subprocess.Popen(
            [
                inference_py,
                "-u",  # unbuffered stdout/stderr so wrap_gptq prints show up in server_log
                inference_script,
                "--model_path",
                model_path,
                "--server",
                "--data_config",
                data_config,
                "--denoising-steps",
                str(denoising_steps),
                "--port",
                str(port),
                "--embodiment-tag",
                "new_embodiment",
                *inference_extra_args,
            ],
            cwd=inference_cwd,
            env=base_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )

    try:
        wait_for_port(port, proc=server_proc)
        eval_cmd = [
            eval_py,
            eval_script,
            "--task_suite_name",
            task_suite,
            "--num_trials_per_task",
            str(num_trials),
            "--num_steps_wait",
            str(num_steps_wait),
            "--port",
            str(port),
            "--task_ids",
            *[str(task_id) for task_id in task_ids],
            "--log_dir",
            str(eval_log_dir),
            "--log_suffix",
            f"shard_{shard_idx}",
            "--rollout_dir",
            str(rollout_dir),
            "--summary_json",
            str(summary_json_path),
            "--init-offset",
            str(init_offset),
            *eval_extra_args,
        ]
        if headless:
            eval_cmd.append("--headless")
        # Saving rollout videos can be enabled per-shard via env var so we can
        # inspect specific failure modes (default off to keep disk usage low).
        if os.environ.get("GR00T_SAVE_VIDEOS", "0") not in ("0", "false", "False"):
            eval_cmd.append("--save-videos")
        else:
            eval_cmd.append("--no-save-videos")

        with open(eval_stdout_log_path, "w") as eval_log:
            subprocess.run(
                eval_cmd,
                cwd=eval_cwd,
                env=base_env,
                stdout=eval_log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=30)


def merge_summaries(
    output_root: Path,
    task_suite: str,
    model_path: str,
    data_config: str,
    gpu_list: list[str],
    benchmark_label: str,
    wbits: int,
    abits: int,
    num_trials: int,
    num_steps_wait: int,
) -> None:
    summary_files = sorted((output_root / "summaries").glob("shard_*.json"))
    if not summary_files:
        raise RuntimeError("No shard summary JSON files found")

    summaries = [json.loads(path.read_text()) for path in summary_files]
    total_episodes = sum(item["total_episodes"] for item in summaries)
    total_successes = sum(item["total_successes"] for item in summaries)
    task_rows = []
    for item in summaries:
        task_rows.extend(item.get("task_summaries", []))
    task_rows.sort(key=lambda row: row["task_id"])

    merged = {
        "task_suite_name": task_suite,
        "model_path": model_path,
        "data_config": data_config,
        "gpu_list": gpu_list,
        "wbits": wbits,
        "abits": abits,
        "num_trials_per_task": num_trials,
        "num_steps_wait": num_steps_wait,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": (float(total_successes) / float(total_episodes)) if total_episodes else 0.0,
        "num_task_shards": len(summaries),
        "task_summaries": task_rows,
        "shard_summaries": summaries,
    }
    (output_root / "merged_summary.json").write_text(json.dumps(merged, indent=2))

    lines = [
        f"# LIBERO {benchmark_label} Summary",
        "",
        f"- Task suite: {task_suite}",
        f"- GPUs: {','.join(gpu_list)}",
        f"- Quantization: W{wbits}A{abits}",
        f"- Benchmark: {benchmark_label}",
        f"- Episodes: {total_successes}/{total_episodes} successes ({merged['total_success_rate'] * 100:.1f}%)",
        "",
        "| task_id | success_rate | successes | episodes | task_description |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task_id']} | {row['success_rate'] * 100:.1f}% | {row['successes']} | {row['episodes']} | {row['task_description']} |"
        )
    (output_root / "merged_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    task_suite = env_str("TASK_SUITE", "libero_10")
    gpu_list = [token.strip() for token in env_str("GPU_LIST", "0,1,2,3,4,5,6,7").split(",") if token.strip()]
    if not gpu_list:
        raise SystemExit("GPU_LIST is empty")

    default_model_path, default_data_config, num_tasks = suite_defaults(task_suite)
    model_path = env_str("MODEL_PATH", default_model_path)
    data_config = env_str("DATA_CONFIG", default_data_config)
    num_trials = env_int("NUM_TRIALS_PER_TASK", 5)
    num_steps_wait = env_int("NUM_STEPS_WAIT", 10)
    denoising_steps = env_int("DENOISING_STEPS", 8)
    wbits = env_int("WBITS", 3)
    abits = env_int("ABITS", 8)
    benchmark_label = env_str("BENCHMARK_LABEL", "DuQuant Benchmark")
    headless = env_str("HEADLESS", "1") == "1"
    port_base = env_int("PORT_BASE", 5600)
    output_root = Path(
        env_str(
            "OUTPUT_ROOT",
            str(
                QUANTVLA_ROOT
                / "results"
                / f"libero_duquant_{task_suite}_w{wbits}a{abits}_gpu{'_'.join(gpu_list)}"
            ),
        )
    )
    packdir = env_str(
        "PACKDIR", str(QUANTVLA_ROOT / f"duquant_packed_paperalign_{task_suite}_w{wbits}a{abits}")
    )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    (output_root / "summaries").mkdir(parents=True, exist_ok=True)
    (output_root / "rollouts").mkdir(parents=True, exist_ok=True)

    # Optional: override the task list (e.g. focus on a single failing task).
    # Format: "5" or "0,2,5" — these task IDs replace the default 0..num_tasks-1.
    # When a single task is overridden onto multiple GPUs, replicate it on each
    # shard and split num_trials across them via per-shard init_offset.
    task_override = os.environ.get("GR00T_TASK_IDS_OVERRIDE", "").strip()
    # Optional global init_offset override (e.g. to run a second 100ep with
    # init_states[10..19] instead of [0..9] per task). All shards share the
    # offset in this case.
    global_init_offset_env = os.environ.get("GR00T_EVAL_INIT_OFFSET", "").strip()
    global_init_offset = int(global_init_offset_env) if global_init_offset_env else 0
    shard_init_offsets: list[int] = [global_init_offset] * len(gpu_list)
    shard_num_trials: list[int] = [num_trials] * len(gpu_list)
    if task_override:
        task_ids_subset = [int(x) for x in task_override.split(",") if x.strip()]
        if len(task_ids_subset) == 1 and len(gpu_list) > 1:
            # Single task → replicate across all shards, split trial range
            tid = task_ids_subset[0]
            base = num_trials // len(gpu_list)
            extra = num_trials % len(gpu_list)
            task_shards = [[tid] for _ in range(len(gpu_list))]
            shard_num_trials = [base + (1 if i < extra else 0) for i in range(len(gpu_list))]
            shard_init_offsets = [sum(shard_num_trials[:i]) for i in range(len(gpu_list))]
            print(f"[multi_gpu] single-task replicate: tid={tid}, "
                  f"trials per shard={shard_num_trials}, offsets={shard_init_offsets}")
        elif len(gpu_list) > len(task_ids_subset):
            # Multi-task replicate: each task gets ~len(gpu_list)/N shards,
            # trials split evenly across replicas. Same algorithm as the
            # default 'auto trial-split' branch but constrained to the
            # overridden task subset.
            n_tasks = len(task_ids_subset)
            base_replicas = len(gpu_list) // n_tasks
            extra = len(gpu_list) - base_replicas * n_tasks
            task_shards = []
            shard_num_trials = []
            shard_init_offsets = []
            for idx, tid in enumerate(task_ids_subset):
                n_replicas = base_replicas + (1 if idx < extra else 0)
                base_trials = num_trials // n_replicas
                rem_trials = num_trials % n_replicas
                cum_offset = 0
                for k in range(n_replicas):
                    nt = base_trials + (1 if k < rem_trials else 0)
                    task_shards.append([tid])
                    shard_num_trials.append(nt)
                    shard_init_offsets.append(global_init_offset + cum_offset)
                    cum_offset += nt
            print(f"[multi_gpu] task override + auto trial-split: {task_ids_subset}, "
                  f"~{len(gpu_list)/n_tasks:.1f} shards/task, num_trials per shard={shard_num_trials[:5]}...")
        else:
            # Multiple tasks → distribute across GPUs (no replication)
            task_shards = [[] for _ in range(len(gpu_list))]
            for idx, tid in enumerate(task_ids_subset):
                task_shards[idx % len(gpu_list)].append(tid)
            print(f"[multi_gpu] task override: {task_ids_subset}, sharded as {task_shards}")
    else:
        if len(gpu_list) > num_tasks:
            # Auto trial-split: more shards than tasks → distribute (task, trial-range)
            # tuples to fully utilize all shards. Each task gets ~len(gpu_list)/num_tasks
            # shards, with trials split evenly across replicas.
            base_replicas = len(gpu_list) // num_tasks
            extra = len(gpu_list) - base_replicas * num_tasks  # first 'extra' tasks get +1 replica
            task_shards = []
            shard_num_trials = []
            shard_init_offsets = []
            for tid in range(num_tasks):
                n_replicas = base_replicas + (1 if tid < extra else 0)
                base_trials = num_trials // n_replicas
                rem_trials = num_trials % n_replicas
                cum_offset = 0
                for k in range(n_replicas):
                    nt = base_trials + (1 if k < rem_trials else 0)
                    task_shards.append([tid])
                    shard_num_trials.append(nt)
                    shard_init_offsets.append(global_init_offset + cum_offset)
                    cum_offset += nt
            print(f"[multi_gpu] auto trial-split: {num_tasks} tasks × ~{len(gpu_list)/num_tasks:.1f} shards/task, "
                  f"num_trials per shard={shard_num_trials[:5]}..., offsets[:5]={shard_init_offsets[:5]}...")
        else:
            task_shards = shard_tasks(num_tasks=num_tasks, num_shards=len(gpu_list))
    (output_root / "task_shards.json").write_text(json.dumps(task_shards, indent=2))

    print("========================================")
    print("LIBERO Benchmark Launcher")
    print("========================================")
    print(f"Benchmark  : {benchmark_label}")
    print(f"Task suite : {task_suite}")
    print(f"Model      : {model_path}")
    print(f"Data config: {data_config}")
    print(f"Output     : {output_root}")
    print(f"GPUs       : {','.join(gpu_list)}")
    print(f"Trials     : {num_trials}")
    print(f"Steps wait : {num_steps_wait}")
    print(f"WBITS/ABITS: {wbits}/{abits}")
    print(f"Tasks      : {num_tasks}")
    print(f"Packdir    : {packdir}")
    print("========================================")

    shard_procs: list[tuple[int, subprocess.Popen]] = []
    shard_driver_logs: list[Path] = []

    for shard_idx, gpu in enumerate(gpu_list):
        task_ids = task_shards[shard_idx]
        if not task_ids:
            print(f"[Skip] shard={shard_idx} gpu={gpu} has no tasks")
            continue

        port = port_base + shard_idx
        shard_trials = shard_num_trials[shard_idx]
        shard_offset = shard_init_offsets[shard_idx]
        driver_log_path = output_root / "logs" / f"driver_shard_{shard_idx}.log"
        shard_driver_logs.append(driver_log_path)
        cmd = [
            GROOT_PY,
            str(Path(__file__).resolve()),
            "--run-shard",
            str(shard_idx),
            gpu,
            str(port),
            json.dumps(task_ids),
            task_suite,
            model_path,
            data_config,
            str(shard_trials),
            str(num_steps_wait),
            str(denoising_steps),
            str(wbits),
            str(abits),
            str(output_root),
            packdir,
            "1" if headless else "0",
            str(shard_offset),
        ]
        with open(driver_log_path, "w") as driver_log:
            proc = subprocess.Popen(cmd, cwd=str(QUANTVLA_ROOT), stdout=driver_log, stderr=subprocess.STDOUT)
        shard_procs.append((shard_idx, proc))
        print(f"[Launch] shard={shard_idx} gpu={gpu} port={port} tasks={task_ids}")
        # Stagger shard launches to avoid simultaneous EGL/mujoco context init,
        # which can fail with EGL_NOT_INITIALIZED when many shards spin up at once.
        # Two modes:
        #   GR00T_SHARD_STAGGER_MODE=fixed (default): sleep STAGGER_SEC seconds.
        #   GR00T_SHARD_STAGGER_MODE=wait_eval: poll the eval log of shard_idx
        #     until it contains "Task:" or "episode" or hits a max timeout. This
        #     serializes EGL init across shards.
        stagger_mode = os.environ.get("GR00T_SHARD_STAGGER_MODE", "fixed")
        if stagger_mode == "wait_eval":
            eval_log_dir = output_root / "logs" / "eval"
            eval_log_glob = f"libero_eval_{task_suite}_shard_{shard_idx}.log"
            wait_timeout = float(os.environ.get("GR00T_SHARD_WAIT_EVAL_TIMEOUT_S", "120"))
            wait_started = time.time()
            print(f"[Launch] waiting for shard={shard_idx} eval to init (max {wait_timeout}s)")
            while time.time() - wait_started < wait_timeout:
                eval_logs = list(eval_log_dir.glob(eval_log_glob))
                if eval_logs:
                    try:
                        content = eval_logs[0].read_text(errors="ignore")
                        if "Task:" in content or "episode" in content.lower():
                            break
                    except Exception:
                        pass
                time.sleep(2)
            else:
                print(f"[Launch] shard={shard_idx} eval did not init within timeout; proceeding")
            print(f"[Launch] shard={shard_idx} eval started after {time.time()-wait_started:.1f}s")
        else:
            stagger_s = float(os.environ.get("GR00T_SHARD_STAGGER_SEC", "5"))
            if stagger_s > 0:
                time.sleep(stagger_s)

    try:
        for shard_idx, proc in shard_procs:
            return_code = proc.wait()
            if return_code != 0:
                print_shard_log_tails(output_root, shard_idx, task_suite)
                raise RuntimeError(f"Shard process exited with code {return_code}")
    except KeyboardInterrupt:
        for _, proc in shard_procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        raise

    merge_summaries(
        output_root=output_root,
        task_suite=task_suite,
        model_path=model_path,
        data_config=data_config,
        gpu_list=gpu_list,
        benchmark_label=benchmark_label,
        wbits=wbits,
        abits=abits,
        num_trials=num_trials,
        num_steps_wait=num_steps_wait,
    )

    print("========================================")
    print("Finished")
    print(f"Output     : {output_root}")
    print(f"Merged JSON: {output_root / 'merged_summary.json'}")
    print(f"Merged MD  : {output_root / 'merged_summary.md'}")
    print(f"Logs       : {output_root / 'logs'}")
    print("========================================")
    return 0


def shard_entry(argv: list[str]) -> int:
    if len(argv) not in (16, 17):
        raise SystemExit(f"Expected 16-17 shard arguments, got {len(argv)}")
    init_offset = 0
    if len(argv) == 17:
        (
            shard_idx,
            gpu,
            port,
            task_ids_json,
            task_suite,
            model_path,
            data_config,
            num_trials,
            num_steps_wait,
            denoising_steps,
            wbits,
            abits,
            output_root,
            packdir,
            headless,
            init_offset_str,
        ) = argv[1:]
        init_offset = int(init_offset_str)
    else:
        (
            shard_idx,
            gpu,
            port,
            task_ids_json,
            task_suite,
            model_path,
            data_config,
            num_trials,
            num_steps_wait,
            denoising_steps,
            wbits,
            abits,
            output_root,
            packdir,
            headless,
        ) = argv[1:]
    run_shard(
        shard_idx=int(shard_idx),
        gpu=gpu,
        port=int(port),
        task_ids=json.loads(task_ids_json),
        task_suite=task_suite,
        model_path=model_path,
        data_config=data_config,
        num_trials=int(num_trials),
        num_steps_wait=int(num_steps_wait),
        denoising_steps=int(denoising_steps),
        wbits=int(wbits),
        abits=int(abits),
        output_root=Path(output_root),
        packdir=packdir,
        headless=headless == "1",
        init_offset=init_offset,
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-shard":
        raise SystemExit(shard_entry(sys.argv[1:]))
    raise SystemExit(main())
