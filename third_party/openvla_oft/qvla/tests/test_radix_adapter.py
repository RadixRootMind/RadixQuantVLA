import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("radix_oft_cli", ROOT / "tools/qvla/oft_cli.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_verified_cache_reuse_and_stale_cache_rejection(tmp_path, monkeypatch):
    args = SimpleNamespace(output_root=tmp_path, backend_root=tmp_path)
    output = tmp_path / "artifact.txt"
    calls = []

    def child(*args, **kwargs):
        calls.append(args)
        output.write_text("verified artifact")

    monkeypatch.setattr(adapter.subprocess, "run", child)
    adapter.stage(args, "test", {"checkpoint": "A"}, output, ["test"])
    adapter.stage(args, "test", {"checkpoint": "A"}, output, ["test"])
    assert len(calls) == 1
    with pytest.raises(ValueError, match="stale"):
        adapter.stage(args, "test", {"checkpoint": "B"}, output, ["test"])
    output.write_text("corrupted")
    with pytest.raises(ValueError, match="stale"):
        adapter.stage(args, "test", {"checkpoint": "A"}, output, ["test"])


def test_legacy_unverified_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "proxy.pt"
    artifact.write_bytes(b"old untracked data")
    with pytest.raises(ValueError, match="Unverified"):
        adapter.stage(SimpleNamespace(output_root=tmp_path), "proxy", {}, artifact, ["unused"])


def test_legacy_cli_dispatches_supported_backends(tmp_path, monkeypatch):
    backend = tmp_path / "backend"
    head = backend / "prismatic/models/action_heads.py"
    head.parent.mkdir(parents=True)
    head.write_text("OFT marker")
    marker = backend / "qvla/calibrate_actions.py"
    marker.parent.mkdir()
    marker.write_text("Supported route marker")
    calls = []
    monkeypatch.setattr(adapter, "main", calls.append)
    monkeypatch.setattr("sys.argv", ["proxy", "--backend-root", str(backend)])
    assert adapter.maybe_dispatch("proxy")
    assert calls[0][:2] == ["--stage", "proxy"]
    head.unlink()
    marker.unlink()
    assert not adapter.maybe_dispatch("proxy")


def test_launcher_oft_plan_uses_training_recipe_and_no2bit(tmp_path):
    env = dict(os.environ)
    for key in (
        "AWESOME_QVLA_ROOT",
        "QUANTVLA_ROOT",
        "BITS",
        "PROXY_BITS",
        "MAX_SAMPLES",
        "OUTPUT_ROOT",
        "PROFILE",
        "METHOD",
        "OPENVLA_BACKEND",
    ):
        env.pop(key, None)
    command = [
        "bash",
        str(ROOT / "scripts/run_openvla_qvla.sh"),
        "--backend",
        "openvla_oft",
        "--action",
        "plan",
        "--output-root",
        str(tmp_path / "run"),
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout[result.stdout.rfind("\n{") + 1 :])
    assert plan["samples"] == 512
    assert plan["bits"] == "0,4,8,16"
    assert plan["max_prune_ratio"] == 0.01


def test_full_precision_cli_preserves_seed_and_wait(monkeypatch):
    from qvla import run_eval

    calls = []
    monkeypatch.setattr(run_eval, "evaluate", calls.append)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--full_precision",
            "--pretrained_checkpoint",
            "checkpoint",
            "--local_log_dir",
            "out",
            "--seed",
            "19",
            "--num_steps_wait",
            "13",
        ],
    )
    run_eval.main()
    assert calls[0].full_precision and calls[0].gates_path is None
    assert calls[0].seed == 19 and calls[0].num_steps_wait == 13
