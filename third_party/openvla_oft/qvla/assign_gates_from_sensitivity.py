"""Global marginal adjacent-demotion allocation from per-channel action sensitivity."""

import argparse
import heapq
import json
import math
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qvla.artifacts import file_hash, write_json
from qvla.quantization import SUPPORTED_BITS


def load_proxies(path, bits):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Empty sensitivity map")
    proxies = {}
    for name, values in raw.items():
        if any(f"proxy_{bit}" not in values for bit in bits):
            raise ValueError(f"Missing requested sensitivity bit in {name}")
        proxies[name] = {bit: values[f"proxy_{bit}"] for bit in bits}
    return proxies


def greedy_allocate(proxies, bit_list, target_avg, max_prune_ratio=0.01):
    bits = sorted(set(bit_list), reverse=True)
    if (
        len(bits) < 2
        or set(bits) - set(SUPPORTED_BITS)
        or bits[0] != 16
        or not math.isfinite(target_avg)
        or not bits[-1] <= target_avg <= bits[0]
    ):
        raise ValueError("Invalid candidate bits or unattainable target")
    if max_prune_ratio is not None and (not math.isfinite(max_prune_ratio) or not 0 <= max_prune_ratio <= 1):
        raise ValueError("Invalid pruning cap")
    if not proxies:
        raise ValueError("Empty sensitivity map")
    assignment, heap, pruned = {}, [], Counter()
    total_channels = 0
    for name, scores in proxies.items():
        if any(bit not in scores for bit in bits):
            raise ValueError(f"Missing sensitivity for {name}")
        values = [scores[bit] for bit in bits]
        if (
            any(
                value.ndim != 1 or not value.numel() or not torch.isfinite(value).all() or (value < 0).any()
                for value in values
            )
            or len({value.numel() for value in values}) != 1
        ):
            raise ValueError(f"Invalid sensitivity shape/value: {name}")
        assignment[name] = [bits[0]] * values[0].numel()
        total_channels += values[0].numel()
    total_bits = bits[0] * total_channels
    serial = 0

    def enqueue(name, index, stage):
        nonlocal serial
        if stage + 1 == len(bits):
            return
        high, low = bits[stage : stage + 2]
        # Incremental error, not absolute low-bit error, per bit saved.
        cost = (float(proxies[name][low][index]) - float(proxies[name][high][index])) / (high - low)
        heapq.heappush(heap, (cost, serial, name, index, stage))
        serial += 1

    for name, values in assignment.items():
        for index in range(len(values)):
            enqueue(name, index, 0)
    while heap and total_bits > target_avg * total_channels:
        _, _, name, index, stage = heapq.heappop(heap)
        high, low = bits[stage : stage + 2]
        if assignment[name][index] != high:
            continue
        if low == 0 and max_prune_ratio is not None and pruned[name] + 1 > int(len(assignment[name]) * max_prune_ratio):
            continue
        assignment[name][index] = low
        total_bits -= high - low
        pruned[name] += low == 0
        enqueue(name, index, stage + 1)
    if total_bits > target_avg * total_channels:
        raise ValueError("Budget cannot be reached under the pruning cap")
    histogram = Counter(bit for values in assignment.values() for bit in values)
    return assignment, {
        "target_avg_bits": target_avg,
        "final_avg_bits": total_bits / total_channels,
        "total_channels": total_channels,
        "initial_total_bits": bits[0] * total_channels,
        "final_total_bits": total_bits,
        "bit_hist": dict(sorted(histogram.items())),
        "max_prune_ratio_per_layer": max_prune_ratio,
        "cost": "marginal",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy_pt", type=Path, required=True)
    parser.add_argument("--bits", default="0,4,8,16")
    parser.add_argument("--target_avg_bits", type=float, required=True)
    parser.add_argument("--max_prune_ratio", type=float, default=0.01)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()
    if args.out_json.exists():
        parser.error("out_json already exists; choose a fresh artifact path")
    bits = sorted({int(value) for value in args.bits.split(",")})
    if set(bits) - set(SUPPORTED_BITS):
        parser.error(f"Choose bits from {SUPPORTED_BITS}")
    calibration = None
    report_path = args.proxy_pt.parent / "status.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if (
            report.get("status") != "complete"
            or report.get("completed_samples") != report.get("samples")
            or report.get("proxy_sha256") != file_hash(args.proxy_pt)
        ):
            parser.error("Calibration is incomplete or its proxy checksum changed")
        calibration = {
            key: report[key]
            for key in ["primary_score", "samples", "completed_samples", "active_modules", "total_channels"]
        }
    proxies = load_proxies(args.proxy_pt, bits)
    assignment, stats = greedy_allocate(proxies, bits, args.target_avg_bits, args.max_prune_ratio)
    write_json(
        args.out_json,
        {
            "schema_version": 1,
            "calibration": calibration,
            "bits": bits,
            "assign": assignment,
            "stats": stats,
            "proxy_sha256": file_hash(args.proxy_pt),
            "activation_bits": 16,
            "budget_unit": "mean output-channel bits; excludes protected parameters/metadata",
            "scope": "Weight fake quantization with BF16 operators; no integer acceleration claim",
        },
    )
    print(stats)


if __name__ == "__main__":
    main()
