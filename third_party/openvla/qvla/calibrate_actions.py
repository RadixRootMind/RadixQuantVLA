"""Validated OpenVLA soft-action directional Jacobian surrogate, not an argmax derivative."""

import argparse
import math
import os
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from qvla.artifacts import file_hash, now, protected_hashes, source_hashes, write_json
from qvla.quantization import SUPPORTED_BITS, quantize_rows, target_modules
from qvla.runtime import configure_cuda, load_model, load_processor, load_records


def projected_channel_scores(gradient, perturbation, channel_axis):
    axes = tuple(axis for axis in range(gradient.ndim) if axis != channel_axis)
    return (gradient.float() * perturbation.float()).sum(dim=axes).square()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained_checkpoint", type=Path, required=True)
    p.add_argument("--calib_jsonl", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--max_samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--task_suite_name", default="libero_spatial")
    p.add_argument("--memory_fraction", type=float, default=0.55)
    p.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT"))
    p.add_argument("--attention", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--flash_attn_path")
    args = p.parse_args()
    if args.temperature <= 0 or not math.isfinite(args.temperature):
        p.error("temperature must be finite and positive")
    rows = load_records(args.calib_jsonl, args.max_samples)
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_cuda(args.memory_fraction)
    model = load_model(str(args.pretrained_checkpoint), args.attention, args.libero_root, args.flash_attn_path)
    model.requires_grad_(False)
    processor = load_processor(str(args.pretrained_checkpoint))
    frozen = protected_hashes(model)
    targets = target_modules(model)
    scores = {
        name: {bit: torch.zeros(module.weight.shape[0], device="cuda", dtype=torch.float32) for bit in SUPPORTED_BITS}
        for name, module in targets.items()
    }
    calls = {name: 0 for name in targets}
    metadata = {
        "status": "running",
        "started": now(),
        "samples": len(rows),
        "completed_samples": 0,
        "primary_score": "E[(sum(J_soft^T r * delta_channel))^2], r=Rademacher/sqrt7",
        "calibration_sha256": file_hash(args.calib_jsonl),
        "seed": args.seed,
        "temperature": args.temperature,
        "bits": list(SUPPORTED_BITS),
        "activation_bits": 16,
        "actual_backend": model._qvla_attention_backend,
        "source_hashes": source_hashes(),
        "teacher_forcing_token_checks": [],
        "protected_parameter_hashes": frozen,
        "scope": "Local temperature-softmax action-bin expectation conditioned on generated action tokens; "
        "not the derivative of discrete argmax actions or exact author calibration.",
    }

    def save():
        active = {
            name: {f"proxy_{bit}": (value / calls[name]).detach().cpu() for bit, value in values.items()}
            for name, values in scores.items()
            if calls[name]
        }
        if not active or any(not torch.isfinite(value).all() for values in active.values() for value in values.values()):
            raise RuntimeError("Empty or nonfinite soft-action sensitivity")
        temporary = output / "proxy.tmp"
        torch.save(active, temporary)
        temporary.replace(output / "proxy.pt")
        metadata.update(
            active_modules=len(active),
            total_channels=sum(v["proxy_16"].numel() for v in active.values()),
            backward_calls=dict(calls),
            proxy_sha256=file_hash(output / "proxy.pt"),
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
        )
        write_json(output / "status.json", metadata)

    write_json(output / "status.json", metadata)
    try:
        for index, row in enumerate(rows):
            with Image.open(row["image"]) as image:
                batch = processor(row["text"], image.convert("RGB"), return_tensors="pt").to(
                    "cuda", dtype=torch.bfloat16
                )
            ids = batch["input_ids"]
            if ids[0, -1].item() != 29871:
                ids = torch.cat((ids, ids.new_tensor([[29871]])), dim=1)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    pixel_values=batch["pixel_values"],
                    max_new_tokens=7,
                    do_sample=False,
                    use_cache=True,
                )
            if generated.shape[1] != ids.shape[1] + 7:
                raise RuntimeError("Reference generation ended before seven actions")
            reference_tokens = generated[:, -7:]
            prefix = generated[:, :-1]
            pixels = batch["pixel_values"].detach().requires_grad_(True)
            handles = []
            for name, module in targets.items():

                def forward_hook(module, inputs, out, name=name):
                    if not out.requires_grad:
                        return
                    is_conv = isinstance(module, torch.nn.Conv2d)
                    axis = 1 if is_conv else out.ndim - 1
                    perturbations = {}
                    with torch.no_grad():
                        for bit in SUPPORTED_BITS:
                            if bit == 16:
                                continue
                            if bit == 0:
                                baseline = out.detach().float()
                                if module.bias is None:
                                    delta = -baseline
                                else:
                                    shape = [1] * out.ndim
                                    shape[axis] = module.bias.numel()
                                    delta = module.bias.detach().float().reshape(shape) - baseline
                            else:
                                weight = quantize_rows(module.weight.detach(), bit)
                                if is_conv:
                                    changed = F.conv2d(
                                        inputs[0].detach(),
                                        weight,
                                        module.bias,
                                        module.stride,
                                        module.padding,
                                        module.dilation,
                                        module.groups,
                                    )
                                else:
                                    changed = F.linear(inputs[0].detach(), weight, module.bias)
                                delta = changed.float() - out.detach().float()
                            perturbations[bit] = delta

                    def backward_hook(gradient):
                        calls[name] += 1
                        for bit, delta in perturbations.items():
                            scores[name][bit].add_(projected_channel_scores(gradient, delta, axis))
                        perturbations.clear()

                    out.register_hook(backward_hook)

                handles.append(module.register_forward_hook(forward_hook))
            try:
                result = model(
                    input_ids=prefix,
                    attention_mask=torch.ones_like(prefix),
                    pixel_values=pixels,
                    use_cache=False,
                    return_dict=True,
                )
                logits = result.logits[:, -7:, :].float()
                lower, upper = model.vocab_size - model.config.n_action_bins, model.vocab_size
                action_ids = torch.arange(lower, upper, device="cuda")
                bin_indices = (model.vocab_size - action_ids - 1).clamp(0, len(model.bin_centers) - 1)
                values = torch.as_tensor(model.bin_centers, device="cuda", dtype=torch.float32)[bin_indices]
                probabilities = (logits[:, :, lower:upper] / args.temperature).softmax(dim=-1)
                actions = (probabilities * values).sum(dim=-1)
                generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
                projection = (
                    torch.randint(0, 2, actions.shape, device="cuda", generator=generator) * 2 - 1
                ).float() / math.sqrt(7)
                metadata["teacher_forcing_token_checks"].append(
                    {
                        "sample_index": index,
                        "matching_tokens": int((logits.argmax(-1) == reference_tokens).sum().item()),
                        "tokens": 7,
                    }
                )
                (actions * projection).sum().backward()
            finally:
                for handle in handles:
                    handle.remove()
            metadata["completed_samples"] = index + 1
            print(f"Calibrated soft-action OpenVLA {index + 1}/{len(rows)}", flush=True)
            if index == 0 or (index + 1) % 8 == 0:
                save()
            del result, logits, probabilities, actions, pixels, batch
        if protected_hashes(model) != frozen or source_hashes() != metadata["source_hashes"]:
            raise RuntimeError("Protected weights or source changed during calibration")
        metadata.update(status="complete", finished=now(), protected_parameters_unchanged=True)
        save()
    except Exception as exc:
        metadata.update(status="failed", finished=now(), error=f"{type(exc).__name__}: {exc}")
        write_json(output / "status.json", metadata)
        raise


if __name__ == "__main__":
    main()
