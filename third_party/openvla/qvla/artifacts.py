"""Small artifact helpers shared by calibration, allocation and evaluation."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def tensor_hash(value):
    import torch

    data = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def protected_hashes(model):
    components = [("projector", model.projector), ("lm_head", model.language_model.lm_head)]
    return {
        f"{prefix}.{name}": tensor_hash(value)
        for prefix, component in components
        for name, value in component.named_parameters()
    }


def source_hashes():
    """Record code identities without requiring a Git executable at runtime."""
    source = Path(__file__).resolve().parents[1]
    paths = list((source / "qvla").glob("*.py"))
    paths.extend(
        source / name
        for name in [
            "experiments/robot/libero/run_libero_eval.py",
            "experiments/robot/libero/libero_utils.py",
            "experiments/robot/openvla_utils.py",
            "experiments/robot/robot_utils.py",
            "prismatic/extern/hf/modeling_prismatic.py",
            "prismatic/extern/hf/processing_prismatic.py",
            "prismatic/extern/hf/configuration_prismatic.py",
        ]
    )
    hashes = {str(path.relative_to(source)): file_hash(path) for path in paths}
    from transformers.models.llama import modeling_llama

    hashes["installed_transformers/modeling_llama.py"] = file_hash(modeling_llama.__file__)
    return hashes
