"""Use the bundled official OFT inference implementation with a private checkpoint view."""

import ast
import inspect
import json
import os
import shutil
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from qvla.artifacts import file_hash

SOURCE = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(directory):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def source_directory():
    """The upstream checkpoint code sync scans ./prismatic relative to cwd."""
    return working_directory(SOURCE)


def verify_bidirectional_attention():
    """Reject a stock Transformers install, including one with the same version."""
    import torch
    from transformers import LlamaConfig, LlamaModel

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.0,
        )
        cfg._attn_implementation = "sdpa"
        model = LlamaModel(cfg).eval()
        with torch.no_grad():
            first = model(torch.tensor([[3, 4, 5, 6]]), use_cache=False).last_hidden_state[:, 0]
            second = model(torch.tensor([[3, 4, 5, 7]]), use_cache=False).last_hidden_state[:, 0]
        difference = float((first - second).abs().max())
    if difference <= 1e-7:
        raise RuntimeError(
            "OFT requires moojink/transformers-openvla-oft's bidirectional SDPA. "
            "Install the pinned fork in qvla/requirements.txt; stock Transformers is causal."
        )
    return difference


def bootstrap(libero_root=None):
    sys.path.insert(0, str(SOURCE))
    if libero_root:
        sys.path.insert(0, str(Path(libero_root).expanduser().resolve()))
    # Import inference modules directly, bypassing unrelated training __init__ imports.
    for name in [
        "prismatic",
        "prismatic.models",
        "prismatic.extern",
        "prismatic.extern.hf",
        "prismatic.training",
        "prismatic.vla",
        "prismatic.vla.datasets",
        "prismatic.vla.datasets.rlds",
        "prismatic.vla.datasets.rlds.utils",
    ]:
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
    verify_bidirectional_attention()
    import experiments.robot.openvla_utils as utils

    return utils


def configure_cuda(memory_fraction):
    import torch

    if not 0 < memory_fraction <= 1:
        raise ValueError("GPU memory fraction must be in (0, 1]")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select one GPU with CUDA_VISIBLE_DEVICES before running this command")
    torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)


def stage_checkpoint(checkpoint, destination):
    """Keep upstream metadata writes away from the original merged checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    destination = Path(destination).resolve()
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise ValueError("--pretrained_checkpoint must be a local merged OFT checkpoint directory")
    if not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Expected merged safetensors weights")
    for component in ("action_head", "proprio_projector"):
        matches = [p for p in checkpoint.glob("*.pt") if component in p.name and "checkpoint" in p.name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {component} checkpoint, found {len(matches)}")
    destination.mkdir(parents=True, exist_ok=False)
    for path in checkpoint.iterdir():
        if not path.is_file():
            continue
        if path.suffix in (".safetensors", ".pt", ".bin"):
            (destination / path.name).symlink_to(path)
        elif path.suffix in (".json", ".py", ".model", ".txt"):
            shutil.copy2(path, destination / path.name)
    for name in ("modeling_prismatic.py", "configuration_prismatic.py"):
        shutil.copy2(SOURCE / "prismatic/extern/hf" / name, destination / name)
    return destination


def make_config(checkpoint, suite="libero_spatial", seed=7, trials=50, log_dir=None):
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    return GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name=suite,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=8,
        seed=seed,
        num_trials_per_task=trials,
        local_log_dir=str(log_dir or "out/qvla_eval"),
        use_wandb=False,
    )


def load_components(cfg, utils):
    from experiments.robot.libero.run_libero_eval import check_unnorm_key

    with source_directory():
        model = utils.get_vla(cfg).requires_grad_(False)
        head = utils.get_action_head(cfg, model.llm_dim).requires_grad_(False)
        proprio = utils.get_proprio_projector(cfg, model.llm_dim, 8).requires_grad_(False)
        processor = utils.get_processor(cfg)
        check_unnorm_key(cfg, model)
    validate_model(model)
    return model, head, proprio, processor


def validate_model(model):
    classes = {type(layer.self_attn).__name__ for layer in model.language_model.model.layers}
    if classes != {"LlamaSdpaAttention"}:
        raise RuntimeError(f"Expected the OFT SDPA inference backend, got {sorted(classes)}")


def differentiable_action_function(utils):
    """Remove only the official helper's outer inference_mode context via AST."""
    tree = ast.parse(inspect.getsource(utils.get_vla_action))
    function = tree.body[0]
    scopes = [node for node in function.body if isinstance(node, ast.With)]
    if len(scopes) != 1 or ast.unparse(scopes[0].items[0].context_expr) != "torch.inference_mode()":
        raise RuntimeError("Upstream action helper changed; review its gradient boundary")
    scope = scopes[0]
    index = function.body.index(scope)
    function.body[index : index + 1] = scope.body
    namespace = dict(utils.__dict__)
    exec(compile(ast.fix_missing_locations(tree), utils.__file__, "exec"), namespace)
    return namespace["get_vla_action"]


def load_records(path, max_samples=None):
    import numpy as np

    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if max_samples is not None:
        if max_samples < 1 or max_samples > len(rows):
            raise ValueError("--max_samples must be positive and no larger than the manifest")
        rows = rows[:max_samples]
    if not rows:
        raise ValueError("Empty calibration manifest")
    seen = set()
    for row in rows:
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ValueError("Calibration rows require a nonempty instruction")
        if np.asarray(row.get("state", [])).shape != (8,) or not np.isfinite(row["state"]).all():
            raise ValueError("Calibration rows require finite 8D proprio state")
        identity = (row["suite"], row["episode_index"])
        if identity in seen:
            raise ValueError("Calibration must use distinct training trajectories")
        seen.add(identity)
        for key in ("image", "wrist_image"):
            image = (path.parent / row[key]).resolve()
            if file_hash(image) != row[key + "_sha256"]:
                raise ValueError(f"Calibration image checksum mismatch: {image}")
            row[key] = str(image)
    return rows


def observation(row):
    import numpy as np
    from PIL import Image

    result = {"state": np.asarray(row["state"], dtype=np.float64).copy()}
    for key, output_key in (("image", "full_image"), ("wrist_image", "wrist_image")):
        with Image.open(row[key]) as image:
            result[output_key] = np.asarray(image.convert("RGB"))
    return result


def native_actions(normalized, stats):
    """Match official NumPy float32 arithmetic before float64 action statistics."""
    import torch

    low = torch.tensor(stats["q01"], device=normalized.device, dtype=torch.float64)
    high = torch.tensor(stats["q99"], device=normalized.device, dtype=torch.float64)
    mask = torch.tensor(stats.get("mask", [True] * 7), device=normalized.device, dtype=torch.bool)
    return torch.where(mask, (0.5 * (normalized.float() + 1)).double() * (high - low + 1e-8) + low, normalized.double())
