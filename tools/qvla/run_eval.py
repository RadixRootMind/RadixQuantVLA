import argparse
import os
import sys

import torch


def _default_backend_root() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "third_party", "openvla")


def _resolve_backend_root(path: str | None) -> str:
    backend_root = path or os.environ.get("QVLA_BACKEND_ROOT") or _default_backend_root()
    backend_root = os.path.abspath(os.path.expanduser(backend_root))
    if not os.path.isdir(backend_root):
        raise FileNotFoundError(f"backend_root not found: {backend_root}")
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)
    return backend_root


def _disable_generate_cache(model) -> None:
    """Avoid OpenVLA eager-attention mask length mismatch during generation."""
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    original_generate = model.generate

    def _generate_no_cache(*args, **kwargs):
        kwargs.setdefault("use_cache", False)
        return original_generate(*args, **kwargs)

    model.generate = _generate_no_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_checkpoint", type=str, required=True)
    p.add_argument("--backend-root", type=str, default=None, help="OpenVLA/OpenVLA-OFT backend root")
    p.add_argument(
        "--gates_path",
        type=str,
        required=True,
        help="gates map (.pt or .json), {layer_name: [bits,...]}",
    )
    p.add_argument("--task_suite_name", type=str, default="libero_spatial")
    p.add_argument("--num_trials_per_task", type=int, default=1)
    p.add_argument("--local_log_dir", type=str, default="rollouts_qvla_fakew")
    p.add_argument("--libero_root", type=str, default=os.environ.get("LIBERO_ROOT", ""))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model-family", type=str, default="openvla")
    p.add_argument("--center-crop", type=str, default="True")
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--run-id-note", type=str, default="qvla_fakew")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    openvla_src_root = _resolve_backend_root(args.backend_root)
    if args.libero_root:
        if os.path.isdir(args.libero_root) and args.libero_root not in sys.path:
            sys.path.insert(0, args.libero_root)

    from inject_fake_w import inject_qvla_weight_fake_quant
    import experiments.robot.libero.run_libero_eval as L
    from experiments.robot.libero.run_libero_eval import eval_libero
    import experiments.robot.robot_utils as R

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _orig_get_model = R.get_model
    _orig_L_get_model = getattr(L, "get_model", None)

    def _patched_get_model(cfg_in):
        model = _orig_get_model(cfg_in)
        _disable_generate_cache(model)
        injected = inject_qvla_weight_fake_quant(
            model,
            gates_path=args.gates_path,
            device=device,
        )
        print(f"[qvla][fake-w] injected into {injected} modules")
        return model

    R.get_model = _patched_get_model  # type: ignore
    if _orig_L_get_model is not None:
        L.get_model = _patched_get_model  # type: ignore

    try:
        _argv_bak = list(sys.argv)
        sys.argv = [
            sys.argv[0],
            "--model_family",
            str(args.model_family),
            "--pretrained_checkpoint",
            str(args.pretrained_checkpoint),
            "--task_suite_name",
            str(args.task_suite_name),
            "--num_trials_per_task",
            str(args.num_trials_per_task),
            "--local_log_dir",
            str(args.local_log_dir),
            "--center_crop",
            str(args.center_crop),
            "--seed",
            str(args.seed),
            "--num_steps_wait",
            str(args.num_steps_wait),
            "--run_id_note",
            str(args.run_id_note),
        ]
        os.makedirs(args.local_log_dir, exist_ok=True)
        eval_libero()
    finally:
        R.get_model = _orig_get_model  # type: ignore
        if _orig_L_get_model is not None:
            L.get_model = _orig_L_get_model  # type: ignore
        sys.argv = _argv_bak


if __name__ == "__main__":
    from oft_cli import maybe_dispatch
    if not maybe_dispatch("eval"):
        main()
