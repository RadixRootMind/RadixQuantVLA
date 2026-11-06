"""Recheck a completed run's canonical log, optional Rich stdout and artifact checksums on CPU."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qvla.artifacts import file_hash
from qvla.evaluation_logs import parse_evaluation


def verify(directory, stdout_path=None):
    directory = Path(directory).resolve()
    state = json.loads((directory / "run.json").read_text())
    if state["status"] != "complete":
        raise ValueError("The evaluator did not complete successfully")
    canonical = directory / state["canonical_log"]
    episodes = directory / "verified_episodes.json"
    if file_hash(canonical) != state["canonical_log_sha256"] or file_hash(episodes) != state["verified_episodes_sha256"]:
        raise ValueError("Evaluation artifacts changed after completion")
    rows = parse_evaluation(canonical.read_text(), state["tasks"], state["trials_per_task"], state["suite"])
    if rows != json.loads(episodes.read_text()):
        raise ValueError("Episode record and canonical log disagree")
    if stdout_path:
        if rows != parse_evaluation(
            Path(stdout_path).read_text(), state["tasks"], state["trials_per_task"], state["suite"]
        ):
            raise ValueError("Canonical and stdout logs disagree")
    wins = sum(row["success"] for row in rows)
    if state["episodes"] != len(rows) or state["successes"] != wins or abs(state["sr"] - wins / len(rows)) > 1e-12:
        raise ValueError("Final metadata does not match the episode records")
    return {"status": "passed", "episodes": len(rows), "successes": wins, "sr": wins / len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--stdout_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.stdout_path), indent=2))


if __name__ == "__main__":
    main()
