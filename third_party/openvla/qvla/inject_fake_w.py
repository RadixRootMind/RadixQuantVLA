"""Strict OpenVLA injection and optional dequantized BF16 checkpoint export."""

import argparse
import os
import shutil
from pathlib import Path

from qvla.quantization import apply_gates


def inject_qvla_weight_fake_quant(model, gates_path, device=None):
    return apply_gates(model, gates_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained_checkpoint", type=Path, required=True)
    p.add_argument("--gates_path", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--attention", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--memory_fraction", type=float, default=0.42)
    p.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT"))
    p.add_argument("--flash_attn_path")
    args = p.parse_args()
    from qvla.artifacts import file_hash, protected_hashes, write_json
    from qvla.runtime import configure_cuda, load_model, load_processor

    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    configure_cuda(args.memory_fraction)
    model = load_model(str(args.pretrained_checkpoint), args.attention, args.libero_root, args.flash_attn_path)
    processor = load_processor(str(args.pretrained_checkpoint))
    before = protected_hashes(model)
    count = apply_gates(model, args.gates_path)
    if protected_hashes(model) != before:
        raise RuntimeError("Protected OpenVLA components changed")
    model.save_pretrained(output)
    processor.save_pretrained(output)
    shutil.copy2(args.pretrained_checkpoint / "dataset_statistics.json", output / "dataset_statistics.json")
    write_json(
        output / "qvla_export.json",
        {
            "quantized_modules": count,
            "activation_bits": 16,
            "gates_sha256": file_hash(args.gates_path),
            "protected_parameters_sha256": before,
            "scope": "BF16 fake-weight export; no packed integer storage or speedup",
        },
    )


if __name__ == "__main__":
    main()
