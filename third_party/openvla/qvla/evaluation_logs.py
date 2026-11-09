"""Strict counter validation for OpenVLA's canonical logs (which have no final totals)."""

import re


def parse_evaluation(text, tasks, trials=50, suite="libero_spatial"):
    if re.search(r"Caught exception:|Episode error:|Traceback \(most recent call last\)|RuntimeError:", text):
        raise ValueError("Runtime error in evaluation log")
    if f"Task suite: {suite}" not in text:
        raise ValueError("Wrong suite")
    chunks = text.split("Task: ")[1:]
    if len(chunks) != len(tasks) * trials:
        raise ValueError("Incomplete evaluation")
    rows, wins = [], 0

    def one(pattern, value):
        matches = re.findall(pattern, value)
        if len(matches) != 1:
            raise ValueError("Missing or duplicated log event")
        return matches[0]

    for index, chunk in enumerate(chunks):
        task, separator, rest = chunk.partition("Starting episode ")
        if not separator or task.strip() != tasks[index // trials]:
            raise ValueError("Task order mismatch")
        trial = int(one(r"^(\d+)\.\.\.", rest))
        success = one(r"\bSuccess: (True|False)\b", rest) == "True"
        completed = int(one(r"# episodes completed so far: (\d+)", rest))
        cumulative, percent = one(r"# successes: (\d+) \(([\d.]+)%\)", rest)
        video = one(r"Saved rollout MP4 at path (\S+\.mp4)", rest)
        wins += success
        if (trial, completed, int(cumulative)) != (index % trials + 1, index + 1, wins):
            raise ValueError("Episode counter mismatch")
        if abs(float(percent) - wins / completed * 100) > 0.050001:
            raise ValueError("Rounded cumulative rate mismatch")
        rows.append(
            dict(
                task=task.strip(),
                trial=trial,
                success=success,
                completed=completed,
                cumulative_successes=wins,
                video=video,
            )
        )
        task_rates = re.findall(r"Current task success rate: ([\d.]+)", rest)
        total_rates = re.findall(r"Current total success rate: ([\d.]+)", rest)
        if completed % trials == 0:
            if len(task_rates) != 1 or len(total_rates) != 1:
                raise ValueError("Missing task summary")
            if abs(float(task_rates[0]) - sum(row["success"] for row in rows[-trials:]) / trials) > 1e-12:
                raise ValueError("Task summary mismatch")
            if abs(float(total_rates[0]) - wins / completed) > 1e-12:
                raise ValueError("Total summary mismatch")
        elif task_rates or total_rates:
            raise ValueError("Premature task summary")
    return rows
