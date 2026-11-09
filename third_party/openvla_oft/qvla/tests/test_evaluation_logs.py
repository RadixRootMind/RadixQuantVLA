import json
import textwrap

import pytest

from qvla.artifacts import file_hash
from qvla.evaluation_logs import parse_evaluation
from qvla.verify_eval import verify

TASKS = ["move the small object to the left tray", "move the large object to the right tray"]


def messages():
    result = ["Task suite: libero_spatial", "Using default initial states"]
    outcomes = [True, False, True, True, False, True]
    wins = 0
    for index, success in enumerate(outcomes):
        wins += success
        result.extend(
            [
                f"Task: {TASKS[index // 3]}",
                f"Starting episode {index % 3 + 1}...",
                f"Saved rollout MP4 at path ./rollouts/example_{index}.mp4",
                f"Success: {success}",
                f"# episodes completed so far: {index + 1}",
                f"# successes: {wins} ({wins / (index + 1) * 100:.1f}%)",
            ]
        )
        if (index + 1) % 3 == 0:
            result.extend(
                [
                    f"Current task success rate: {sum(outcomes[index - 2 : index + 1]) / 3}",
                    f"Current total success rate: {wins / (index + 1)}",
                ]
            )
    result.extend(["Final results:", "Total episodes: 6", "Total successes: 4", "Overall success rate: 0.6667 (66.7%)"])
    return result


def rich_stdout():
    lines = []
    for message in messages():
        if message.startswith("Saved rollout"):
            lines.append(message)
            continue
        wrapped = textwrap.wrap(message, width=26)
        lines.append("\r 0%| | 0/3\x1b[A INFO | >> " + wrapped[0] + "   run_libero_eval.py:221")
        lines.extend("                          " + line for line in wrapped[1:])
    return "\n".join(lines)


def test_actual_rich_wrapping_and_canonical_log_agree():
    rows = parse_evaluation("\n".join(messages()), TASKS, trials=3)
    assert rows == parse_evaluation(rich_stdout(), TASKS, trials=3)
    assert len(rows) == 6 and sum(row["success"] for row in rows) == 4


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text + "\nEpisode error: simulator crashed",
        lambda text: text.replace("Success: True", "Success: False", 1),
        lambda text: text.replace("# episodes completed so far: 1", "# episodes completed so far: 2", 1),
        lambda text: text.replace(TASKS[0], TASKS[1], 1),
        lambda text: text.replace("Total episodes: 6", "Total episodes: 5"),
        lambda text: text.replace("Overall success rate: 0.6667", "Overall success rate: 0.7777"),
        lambda text: text.replace("Starting episode 1...", "Starting episode 2...", 1),
    ],
)
def test_incomplete_or_inconsistent_logs_are_rejected(mutate):
    with pytest.raises(ValueError):
        parse_evaluation(mutate("\n".join(messages())), TASKS, trials=3)


def test_offline_verification_detects_artifact_changes(tmp_path):
    log = tmp_path / "EVAL-example.txt"
    log.write_text("\n".join(messages()))
    episodes = tmp_path / "verified_episodes.json"
    episodes.write_text(json.dumps(parse_evaluation(log.read_text(), TASKS, 3)))
    state = {
        "status": "complete",
        "tasks": TASKS,
        "trials_per_task": 3,
        "suite": "libero_spatial",
        "episodes": 6,
        "successes": 4,
        "sr": 4 / 6,
        "canonical_log": log.name,
        "canonical_log_sha256": file_hash(log),
        "verified_episodes_sha256": file_hash(episodes),
    }
    (tmp_path / "run.json").write_text(json.dumps(state))
    assert verify(tmp_path)["successes"] == 4
    log.write_text(log.read_text() + "tampered")
    with pytest.raises(ValueError, match="artifacts changed"):
        verify(tmp_path)
