"""Explicit causal OpenVLA loading, separate from OFT's bidirectional environment."""

import importlib
import json
import os
import sys
import types
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
ATTENTION_CLASSES = {
    "eager": "LlamaAttention",
    "sdpa": "LlamaSdpaAttention",
    "flash_attention_2": "LlamaFlashAttention2",
}


def bootstrap(libero_root=None):
    sys.path.insert(0, str(SOURCE))
    if libero_root:
        sys.path.insert(0, str(Path(libero_root).expanduser().resolve()))
    for name in ("prismatic", "prismatic.extern", "prismatic.extern.hf"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(SOURCE / Path(*name.split(".")))]
            sys.modules[name] = module
    import tensorflow as tf
    import torch

    tf.config.set_visible_devices([], "GPU")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def activate_flash(path=None):
    path = path or os.environ.get("OPENVLA_FLASH_ATTN_PATH")
    if path:
        sys.path.insert(0, str(Path(path).expanduser().resolve()))
    import flash_attn

    if not flash_attn.__version__.startswith("2.5.5"):
        raise RuntimeError("Validated OpenVLA FA2 profiles require flash-attn2.5.5")
    return {"version": flash_attn.__version__, "file": str(Path(flash_attn.__file__).resolve())}


def verify_causal_attention(attention):
    import torch
    from transformers import LlamaConfig, LlamaModel

    if attention not in ATTENTION_CLASSES:
        raise ValueError("Unsupported attention backend")
    if attention == "flash_attention_2":
        return
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(7)
        cfg = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.0,
        )
        cfg._attn_implementation = attention
        model = LlamaModel(cfg).eval()
        with torch.no_grad():
            a = model(torch.tensor([[3, 4, 5, 6]]), use_cache=False).last_hidden_state[:, 0]
            b = model(torch.tensor([[3, 4, 5, 7]]), use_cache=False).last_hidden_state[:, 0]
        if not torch.equal(a, b):
            raise RuntimeError(
                "OpenVLA requires causal attention: use stock Transformers4.40.1 in a separate "
                "environment. OFT's bidirectional SDPA fork is incompatible with this profile."
            )


def load_model(checkpoint, attention="sdpa", libero_root=None, flash_path=None):
    if attention == "flash_attention_2":
        activate_flash(flash_path)
    bootstrap(libero_root)
    verify_causal_attention(attention)
    import torch

    configuration = importlib.import_module("prismatic.extern.hf.configuration_prismatic")
    modeling = importlib.import_module("prismatic.extern.hf.modeling_prismatic")
    cfg = configuration.OpenVLAConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg._attn_implementation = attention
    cfg.text_config._attn_implementation = attention
    model = (
        modeling.OpenVLAForActionPrediction.from_pretrained(
            checkpoint,
            config=cfg,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    classes = sorted({type(layer.self_attn).__name__ for layer in model.language_model.model.layers})
    if classes != [ATTENTION_CLASSES[attention]]:
        raise RuntimeError(f"Attention selection failed: requested={attention}, actual={classes}")
    stats = Path(checkpoint) / "dataset_statistics.json"
    if stats.is_file():
        model.norm_stats = json.loads(stats.read_text())
    model._qvla_attention_backend = {
        "requested": attention,
        "actual_classes": classes,
        "language_layers": len(model.language_model.model.layers),
    }
    return model


def load_processor(checkpoint):
    from transformers import AutoTokenizer

    processing = importlib.import_module("prismatic.extern.hf.processing_prismatic")
    image = processing.PrismaticImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    return processing.PrismaticProcessor(image, tokenizer)


def configure_cuda(fraction):
    import torch

    if torch.cuda.device_count() != 1 or not 0 < fraction <= 1:
        raise ValueError("Select one CUDA GPU and memory_fraction in(0,1]")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)


def load_records(path, count=None):
    from qvla.artifacts import file_hash

    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if count is not None:
        if not 1 <= count <= len(rows):
            raise ValueError("Requested calibration prefix is missing")
        rows = rows[:count]
    if not rows:
        raise ValueError("Empty calibration manifest")
    for row in rows:
        if row.get("preprocessed") is not True:
            raise ValueError("Use prepared OpenVLA calibration images with preprocessed=true")
        image = (path.parent / row["image"]).resolve()
        if file_hash(image) != row["image_sha256"]:
            raise ValueError("Calibration image checksum mismatch")
        row["image"] = str(image)
        if not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError("OpenVLA calibration requires a prepared text prompt")
    return rows
