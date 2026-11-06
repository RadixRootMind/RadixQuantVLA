import itertools
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from qvla.assign_gates_from_sensitivity import restore_pruned_channels
from qvla.calibrate_actions import projected_channel_scores
from qvla.evaluation_logs import parse_evaluation
from qvla.runtime import verify_causal_attention


def test_directional_proxy_matches_explicit_jacobian():
    x = torch.tensor([[0.2, -0.3], [0.7, 1.0]], dtype=torch.float64)
    delta = torch.tensor([[0.1, 0.3], [-0.4, 0.2]], dtype=torch.float64)

    def actions(value):
        return torch.stack((value[0, 0] + 2 * value[1, 1], value[0, 1] - 3 * value[1, 0]))

    jacobian = torch.autograd.functional.jacobian(actions, x)
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=2):
        projection = torch.tensor(signs, dtype=torch.float64) / 2**0.5
        gradient = torch.einsum("a,atc->tc", projection, jacobian)
        values.append(projected_channel_scores(gradient, delta, 1))
    expected = (jacobian * delta).sum(1).square().sum(0) / 2
    torch.testing.assert_close(torch.stack(values).mean(0).double(), expected, rtol=1e-6, atol=1e-7)


def test_pruning_repair_conserves_bits_and_restores_highest_benefit():
    scores = {
        "layer": {
            0: torch.tensor([10.0, 1.0, 9.0, 8.0]),
            4: torch.tensor([2.0, 2.0, 2.0, 2.0]),
            8: torch.tensor([1.0, 1.0, 1.0, 1.0]),
            16: torch.zeros(4),
        }
    }
    result, stats = restore_pruned_channels(scores, {"layer": [0, 0, 16, 16]}, 8, 0.5)
    assert result["layer"] == [8, 0, 8, 16]
    assert sum(result["layer"]) == 32 and stats["restored_channels"] == 1


def test_no_pruning_repair_accepts_candidates_without_zero_or_four_bits():
    scores = {"layer": {8: torch.tensor([1.0, 2.0]), 16: torch.zeros(2)}}
    assignment = {"layer": [8, 8]}
    result, stats = restore_pruned_channels(scores, assignment, 8, 0.25)
    assert result == assignment and stats["restored_channels"] == 0


def fixture():
    return (
        "Task suite: libero_spatial\nTask: move the object\nStarting episode 1...\n"
        "Saved rollout MP4 at path ./rollouts/toy.mp4\nSuccess: True\n"
        "# episodes completed so far: 1\n# successes: 1 (100.0%)\n"
        "Current task success rate: 1.0\nCurrent total success rate: 1.0\n"
    )


def test_openvla_logs_need_no_oft_final_totals_and_reject_runtime_errors():
    assert parse_evaluation(fixture(), ["move the object"], 1)[0]["success"]
    for value in (fixture() + "Caught exception: injected", fixture().replace("Success: True", "Success: False")):
        with pytest.raises(ValueError):
            parse_evaluation(value, ["move the object"], 1)


def test_stock_sdpa_is_causal():
    before = torch.get_rng_state().clone()
    verify_causal_attention("sdpa")
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "backend,attention,repair", [("openvla", "flash_attention_2", 0.25), ("openvla_oft", "sdpa", 0.0)]
)
def test_radix_plan_selects_validated_backend_defaults(tmp_path, backend, attention, repair):
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ)
    for name in [
        "AWESOME_QVLA_ROOT",
        "QUANTVLA_ROOT",
        "BITS",
        "PROXY_BITS",
        "MAX_SAMPLES",
        "PROFILE",
        "METHOD",
        "OPENVLA_BACKEND",
        "QVLA_ATTENTION",
        "QVLA_RESTORE_PRUNED_FRACTION",
    ]:
        env.pop(name, None)
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/run_openvla_qvla.sh"),
            "--backend",
            backend,
            "--action",
            "plan",
            "--output-root",
            str(tmp_path / "run"),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(result.stdout[result.stdout.rfind("\n{") + 1 :])
    assert plan["samples"] == 512 and plan["bits"] == "0,4,8,16"
    assert plan["attention"] == attention and plan["restore_pruned_fraction"] == repair
