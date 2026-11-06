"""Strict OFT weight fake quantization and optional self-contained BF16 checkpoint export."""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvla.quantization import apply_gates


def inject_qvla_weight_fake_quant(model, gates_path, device=None):
    """Compatibility API; weights are updated on each module's existing device."""
    return apply_gates(model, gates_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained_checkpoint", type=Path, required=True)
    parser.add_argument("--gates_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--libero_root", default=os.environ.get("LIBERO_ROOT"))
    parser.add_argument("--memory_fraction", type=float, default=0.42)
    args = parser.parse_args()
    from qvla.artifacts import file_hash, protected_hashes, write_json
    from qvla.runtime import bootstrap, configure_cuda, load_components, make_config, stage_checkpoint

    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    configure_cuda(args.memory_fraction)
    utils = bootstrap(args.libero_root)
    with tempfile.TemporaryDirectory(prefix="qvla-oft-export-") as temporary:
        checkpoint = stage_checkpoint(args.pretrained_checkpoint, Path(temporary) / "checkpoint")
        cfg = make_config(checkpoint)
        model, head, proprio, processor = load_components(cfg, utils)
        frozen = protected_hashes(model, head, proprio)
        count = apply_gates(model, args.gates_path)
        if protected_hashes(model, head, proprio) != frozen:
            raise RuntimeError("Quantization changed protected BF16 components")
        model.save_pretrained(args.out_dir)
        processor.save_pretrained(args.out_dir)
        # Keep the original continuous head, proprio projector, statistics and custom code.
        for path in checkpoint.iterdir():
            if path.suffix == ".pt" or path.name in (
                "dataset_statistics.json",
                "modeling_prismatic.py",
                "configuration_prismatic.py",
            ):
                shutil.copy2(path, args.out_dir / path.name)
        write_json(
            args.out_dir / "qvla_export.json",
            {
                "activation_bits": 16,
                "quantized_modules": count,
                "gates_sha256": file_hash(args.gates_path),
                "protected_parameters_sha256": frozen,
                "scope": "Dequantized BF16 weights; exporting does not provide integer storage or speedup.",
            },
        )
    print(f"Exported fake-weight OFT checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
