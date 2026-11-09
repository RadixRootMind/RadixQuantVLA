import json

import pytest

from qvla.artifacts import file_hash
from qvla.runtime import load_records, stage_checkpoint


def test_checkpoint_staging_preserves_original_metadata(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    for name in (
        "config.json",
        "model.safetensors",
        "action_head--1_checkpoint.pt",
        "proprio_projector--1_checkpoint.pt",
    ):
        (original / name).write_text("original bytes")
    staged = stage_checkpoint(original, tmp_path / "runtime")
    (staged / "config.json").write_text("metadata changed by official loader")
    assert (original / "config.json").read_text() == "original bytes"
    assert (staged / "model.safetensors").is_symlink()
    assert (staged / "modeling_prismatic.py").is_file()


def test_calibration_hash_state_and_duplicate_checks(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image payload checksum fixture")
    row = {
        "suite": "libero_spatial",
        "episode_index": 1,
        "step_index": 3,
        "instruction": "move the object",
        "state": [0.0] * 8,
        "image": image.name,
        "image_sha256": file_hash(image),
        "wrist_image": image.name,
        "wrist_image_sha256": file_hash(image),
    }
    manifest = tmp_path / "records.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    assert load_records(manifest)[0]["image"] == str(image)
    manifest.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="distinct training trajectories"):
        load_records(manifest)
    manifest.write_text(json.dumps(row) + "\n")
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_records(manifest)
