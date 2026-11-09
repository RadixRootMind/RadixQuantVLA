"""Calibrate QVLA from native continuous OFT actions on two-camera LIBERO records."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvla.action_scores import directional_score, variance_jacobian_score
from qvla.artifacts import file_hash, now, protected_hashes, source_hashes, tensor_hash, write_json
from qvla.runtime import (
    bootstrap,
    configure_cuda,
    differentiable_action_function,
    load_components,
    load_records,
    make_config,
    native_actions,
    observation,
    stage_checkpoint,
)


def calibrate(args):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from qvla.quantization import SUPPORTED_BITS, apply_gates, quantize_rows, target_modules

    rows = load_records(args.calib_jsonl, args.max_samples)
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_cuda(args.memory_fraction)
    torch.manual_seed(args.seed)
    utils = bootstrap(args.libero_root)
    checkpoint = stage_checkpoint(args.pretrained_checkpoint, output / "runtime_checkpoint")
    cfg = make_config(checkpoint, args.task_suite_name, args.seed)
    model, head, proprio, processor = load_components(cfg, utils)
    frozen = protected_hashes(model, head, proprio)
    modules = target_modules(model)
    scores = {
        name: {bit: torch.zeros(module.weight.shape[0], device="cuda", dtype=torch.float64) for bit in SUPPORTED_BITS}
        for name, module in modules.items()
    }
    directional = {
        name: {bit: torch.zeros_like(value) for bit, value in values.items()} for name, values in scores.items()
    }
    calls = {name: 0 for name in modules}
    quantized_bits = [bit for bit in SUPPORTED_BITS if bit != 16]
    report = {
        "status": "running",
        "started": now(),
        "samples": len(rows),
        "completed_samples": 0,
        "bits": list(SUPPORTED_BITS),
        "activation_bits": 16,
        "seed": args.seed,
        "memory_fraction_cap": args.memory_fraction,
        "action_shape": [8, 7],
        "calibration_sha256": file_hash(args.calib_jsonl),
        "source_hashes": source_hashes(),
        "protected_parameter_hashes": frozen,
        "inference_gradient_output_checks": [],
        "fullshape_identity_calls": 0,
        "active_modules": 0,
        "primary_score": "E[||J^T r||_channel^2 * mean(delta_channel^2)], r=Rademacher/sqrt56",
        "secondary_score": "E[(sum(J^T r * delta_channel))^2]; diagnostic only",
        "scope": "Native8x7 continuous actions before simulator gripper binarization; "
        "AppendixG variance-times-Jacobian proxy. Author-specific forward refinement "
        "and rollout cross-validation are not implemented.",
    }
    write_json(output / "status.json", report)
    # Check the actual injector at16bit before taking any action gradients.
    # This is an offline byte-identity control, not an environment baseline run.
    control = output / "all16_control.json"
    write_json(control, {"assign": {name: [16] * module.weight.shape[0] for name, module in modules.items()}})
    before_control = {name: tensor_hash(module.weight) for name, module in modules.items()}
    controlled_modules = apply_gates(model, control)
    after_control = {name: tensor_hash(module.weight) for name, module in modules.items()}
    if before_control != after_control or protected_hashes(model, head, proprio) != frozen:
        raise RuntimeError("The all16 gate control changed model weights")
    report["all16_gate_control"] = {
        "status": "passed",
        "modules": controlled_modules,
        "all_target_weights_byte_exact": True,
        "protected_components_exact": True,
    }
    write_json(output / "status.json", report)
    gradient_action = differentiable_action_function(utils)

    class GradientProcessor:
        def __call__(self, *positional, **keywords):
            batch = processor(*positional, **keywords)
            batch["pixel_values"].requires_grad_(True)
            return batch

    captured = {}

    def capture_head(module, inputs, value):
        captured["normalized"] = value

    head_hook = head.model.register_forward_hook(capture_head)
    stats = model.get_action_stats(cfg.unnorm_key)
    begin = time.monotonic()
    active_names = None

    def save_scores():
        active = {
            name: {f"proxy_{bit}": (value / report["completed_samples"]).cpu() for bit, value in values.items()}
            for name, values in scores.items()
            if calls[name]
        }
        if not active or any(
            not torch.isfinite(value).all() or (value < 0).any()
            for values in active.values()
            for value in values.values()
        ):
            raise RuntimeError("Empty, negative or nonfinite action sensitivity")
        for filename, values in [
            ("proxy.pt", active),
            (
                "directional.pt",
                {
                    name: {f"proxy_{bit}": (value / report["completed_samples"]).cpu() for bit, value in values.items()}
                    for name, values in directional.items()
                    if calls[name]
                },
            ),
        ]:
            temporary_path = output / (filename + ".tmp")
            torch.save(values, temporary_path)
            temporary_path.replace(output / filename)
        report.update(
            active_modules=len(active),
            calls=dict(calls),
            total_channels=sum(values["proxy_16"].numel() for values in active.values()),
            proxy_sha256=file_hash(output / "proxy.pt"),
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic() - begin,
        )
        write_json(output / "status.json", report)

    try:
        for index, row in enumerate(rows):
            reference = None
            if index < args.reference_checks:
                reference = np.asarray(
                    utils.get_vla_action(cfg, model, processor, observation(row), row["instruction"], head, proprio)
                )
            captured.clear()
            temporary = {}
            handles = []
            for name, module in modules.items():

                def forward(module, inputs, out, name=name, storage=temporary):
                    if not out.requires_grad:
                        raise RuntimeError(f"No differentiable module output: {name}")
                    x = inputs[0].detach()
                    y = out.detach()
                    axis = 1 if isinstance(module, torch.nn.Conv2d) else y.ndim - 1
                    axes = tuple(j for j in range(y.ndim) if j != axis)

                    def backward(gradient):
                        with torch.no_grad():
                            value = storage.setdefault(
                                name,
                                {
                                    "gain": torch.zeros(y.shape[axis], device="cuda", dtype=torch.float64),
                                    "noise": {
                                        bit: torch.zeros(y.shape[axis], device="cuda", dtype=torch.float64)
                                        for bit in quantized_bits
                                    },
                                    "dot": {
                                        bit: torch.zeros(y.shape[axis], device="cuda", dtype=torch.float64)
                                        for bit in quantized_bits
                                    },
                                    "positions": 0,
                                },
                            )
                            if isinstance(module, torch.nn.Conv2d):

                                def apply(weight):
                                    return F.conv2d(
                                        x,
                                        weight,
                                        module.bias,
                                        module.stride,
                                        module.padding,
                                        module.dilation,
                                        module.groups,
                                    )
                            else:

                                def apply(weight):
                                    return F.linear(x, weight, module.bias)

                            if not torch.equal(apply(module.weight), y):
                                raise RuntimeError(f"Full-shape output identity failed: {name}")
                            report["fullshape_identity_calls"] += 1
                            value["gain"].add_(gradient.float().square().sum(dim=axes).double())
                            value["positions"] += y.numel() // y.shape[axis]
                            for bit in quantized_bits:
                                delta = apply(quantize_rows(module.weight, bit)).float() - y.float()
                                value["noise"][bit].add_(delta.square().sum(dim=axes).double())
                                value["dot"][bit].add_((gradient.float() * delta).sum(dim=axes).double())
                            calls[name] += 1

                    out.register_hook(backward)

                handles.append(module.register_forward_hook(forward))
            try:
                returned = np.asarray(
                    gradient_action(cfg, model, GradientProcessor(), observation(row), row["instruction"], head, proprio)
                )
                normalized = captured["normalized"]
                if normalized.shape != (1, 8, 7) or not normalized.requires_grad:
                    raise RuntimeError("Expected differentiable continuous8x7 actions")
                actions = native_actions(normalized, stats)
                if not np.array_equal(actions.detach().cpu().numpy()[0], returned):
                    raise RuntimeError("Differentiable/native action unnormalization mismatch")
                if reference is not None:
                    difference = float(np.max(np.abs(reference - returned)))
                    if difference != 0:
                        raise RuntimeError("Gradient forward differs from official inference")
                    report["inference_gradient_output_checks"].append(
                        {"index": index, "maximum_native_action_difference": difference}
                    )
                generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
                projection = torch.randint(0, 2, actions.shape, device="cuda", generator=generator) * 2 - 1
                projection = projection / math.sqrt(actions.numel())
                (actions * projection).sum().backward()
            finally:
                for handle in handles:
                    handle.remove()
            if not temporary:
                raise RuntimeError("No active action-gradient modules")
            if active_names is None:
                active_names = set(temporary)
            if set(temporary) != active_names:
                raise RuntimeError("Active gradient modules changed between calibration records")
            # A vision module is called once per camera. Aggregate both calls before
            # multiplying gain/noise or squaring the directional perturbation.
            for name, item in temporary.items():
                for bit in quantized_bits:
                    scores[name][bit].add_(variance_jacobian_score(item["gain"], item["noise"][bit], item["positions"]))
                    directional[name][bit].add_(directional_score(item["dot"][bit]))
            report["completed_samples"] = index + 1
            print(f"Calibrated native OFT actions {index + 1}/{len(rows)}", flush=True)
            if index == 0 or (index + 1) % args.save_every == 0:
                save_scores()
            del temporary, normalized, actions, captured["normalized"]
            torch.cuda.empty_cache()
        if protected_hashes(model, head, proprio) != frozen:
            raise RuntimeError("Protected components changed during calibration")
        if source_hashes() != report["source_hashes"]:
            raise RuntimeError("Source files changed during calibration")
        report.update(status="complete", finished=now(), protected_parameters_unchanged=True)
        save_scores()
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=now())
        write_json(output / "status.json", report)
        raise
    finally:
        head_hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", type=Path, required=True)
    parser.add_argument("--calib_jsonl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT"))
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_samples", type=int, help="Use a prefix for explicit smoke checks")
    parser.add_argument("--reference_checks", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=16)
    parser.add_argument("--memory_fraction", type=float, default=0.55)
    args = parser.parse_args()
    if args.save_every < 1 or args.reference_checks < 1:
        parser.error("save_every and reference_checks must be positive")
    calibrate(args)


if __name__ == "__main__":
    main()
