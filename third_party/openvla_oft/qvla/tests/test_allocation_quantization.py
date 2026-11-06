import json

import pytest
import torch
from torch import nn

from qvla.assign_gates_from_sensitivity import greedy_allocate
from qvla.assign_gates_from_sensitivity import main as allocate_main
from qvla.quantization import apply_gates, load_gates, quantize_rows


def test_allocator_uses_marginal_cost_instead_of_absolute_low_bit_error():
    scores = {
        "layer": {
            16: torch.tensor([0.0, 0.0]),
            8: torch.tensor([10.0, 1.0]),
            4: torch.tensor([11.0, 4.0]),
            0: torch.tensor([100.0, 10.0]),
        }
    }
    assignment, stats = greedy_allocate(scores, [0, 4, 8, 16], 4, max_prune_ratio=None)
    assert assignment == {"layer": [4, 4]}
    assert stats["final_avg_bits"] == 4


def test_unreachable_pruning_budget_and_missing_score_fail():
    scores = {"layer": {bit: torch.arange(100, dtype=torch.float64) for bit in (0, 4, 8, 16)}}
    assignment, stats = greedy_allocate(scores, [0, 4, 8, 16], 4, max_prune_ratio=0.01)
    assert assignment["layer"].count(0) <= 1
    assert stats["final_avg_bits"] <= 4
    with pytest.raises(ValueError, match="Budget cannot"):
        greedy_allocate(scores, [0, 4, 8, 16], 3, max_prune_ratio=0)
    del scores["layer"][8]
    with pytest.raises(ValueError, match="Missing sensitivity"):
        greedy_allocate(scores, [0, 4, 8, 16], 4)


def model():
    result = nn.Module()
    result.language_model = nn.Module()
    result.language_model.model = nn.ModuleDict({"first": nn.Linear(3, 3), "second": nn.Linear(3, 3)})
    result.vision_backbone = nn.ModuleDict({"conv": nn.Conv2d(3, 3, kernel_size=2)})
    result.projector = nn.Linear(3, 3)
    return result.to(torch.bfloat16)


def write_gates(path, assignment):
    path.write_text(json.dumps({"assign": assignment, "stats": {"max_prune_ratio_per_layer": 0.5}}))


def test_invalid_gate_fails_before_any_model_mutation(tmp_path):
    instance = model()
    before = {name: value.detach().clone() for name, value in instance.named_parameters()}
    path = tmp_path / "gates.json"
    write_gates(path, {"language_model.model.first": [4, 4, 4], "language_model.model.second": [8]})
    with pytest.raises(ValueError, match="length mismatch"):
        apply_gates(instance, path)
    assert all(
        torch.equal(value.view(torch.uint8), before[name].view(torch.uint8))
        for name, value in instance.named_parameters()
    )


def test_injection_keeps_bias_fp_rows_and_protected_projector(tmp_path):
    instance = model()
    path = tmp_path / "gates.json"
    write_gates(path, {"language_model.model.first": [0, 4, 16], "vision_backbone.conv": [4, 8, 16]})
    first = instance.language_model.model["first"]
    first.weight.data[2, 0] = -0.0
    row = first.weight[2].detach().clone().view(torch.uint8)
    bias = first.bias.detach().clone()
    protected = instance.projector.weight.detach().clone()
    assert apply_gates(instance, path) == 2
    assert torch.count_nonzero(first.weight[0]) == 0
    assert torch.equal(first.weight[2].view(torch.uint8), row)
    assert torch.equal(first.bias, bias)
    assert torch.equal(instance.projector.weight, protected)


@pytest.mark.parametrize("bad", [[2, 4, 8], [4.0, 4, 4], [True, 4, 4]])
def test_bit_assignments_are_strict(tmp_path, bad):
    path = tmp_path / "gates.json"
    write_gates(path, {"language_model.model.first": bad})
    with pytest.raises(ValueError, match="Invalid bit"):
        load_gates(path)


def test_zero_rows_and_conv_rows_are_finite():
    values = torch.zeros(4, 3, 2, 2, dtype=torch.bfloat16)
    for bit in (0, 4, 8, 16):
        result = quantize_rows(values, bit)
        assert torch.isfinite(result).all() and torch.count_nonzero(result) == 0


def test_cli_rejects_partial_calibration_instead_of_allocating(tmp_path, monkeypatch):
    proxy = tmp_path / "proxy.pt"
    torch.save({"layer": {f"proxy_{bit}": torch.ones(3) for bit in (0, 4, 8, 16)}}, proxy)
    (tmp_path / "status.json").write_text(json.dumps({"status": "running", "samples": 8, "completed_samples": 7}))
    monkeypatch.setattr(
        "sys.argv",
        ["allocate", "--proxy_pt", str(proxy), "--target_avg_bits", "4", "--out_json", str(tmp_path / "gates.json")],
    )
    with pytest.raises(SystemExit) as error:
        allocate_main()
    assert error.value.code == 2
    assert not (tmp_path / "gates.json").exists()
