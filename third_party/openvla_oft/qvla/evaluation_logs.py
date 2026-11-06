"""Audit canonical OFT logs or Rich-formatted stdout without unbounded multiline regexes."""

import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(text):
    require(
        not re.search(r"Episode error:|Traceback \(most recent call last\)|RuntimeError:", text),
        "Runtime error in evaluation log",
    )
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    lines = []
    for line in text.replace("\r", "\n").splitlines():
        marker = re.search(r"\bINFO\s+\|\s*>>\s*", line)
        if marker:
            line = line[marker.end() :]
        line = re.sub(r"\s+run_libero_eval\.py:\d+\s*$", "", line)
        lines.append(line.strip())
    return " ".join(" ".join(lines).split())


def one(pattern, text, label):
    matches = re.findall(pattern, text)
    require(len(matches) == 1, f"{label}: expected one match, got {len(matches)}")
    return matches[0]


def parse_evaluation(text, tasks, trials=50, suite="libero_spatial"):
    require(trials > 0 and bool(tasks) and len(set(tasks)) == len(tasks), "Invalid expected task protocol")
    text = normalize(text)
    require(f"Task suite: {suite}" in text, "Wrong task suite")
    require("Using default initial states" in text, "Wrong initial-state protocol")
    expected = len(tasks) * trials
    chunks = text.split("Task: ")[1:]
    require(len(chunks) == expected, f"Expected {expected} episode blocks, got {len(chunks)}")
    rows = []
    wins = 0
    for index, chunk in enumerate(chunks):
        task, separator, rest = chunk.partition("Starting episode ")
        require(separator and task.strip() == tasks[index // trials], f"Task order at {index + 1}")
        trial = int(one(r"^(\d+)\.\.\.", rest, "Trial"))
        success = one(r"\bSuccess: (True|False)\b", rest, "Outcome") == "True"
        completed = int(one(r"# episodes completed so far: (\d+)\b", rest, "Completed counter"))
        cumulative, percent = one(r"# successes: (\d+) \(([\d.]+)%\)", rest, "Success counter")
        video = one(r"Saved rollout MP4 at path (\S+\.mp4)", rest, "Video")
        wins += success
        require(
            (trial, completed, int(cumulative)) == (index % trials + 1, index + 1, wins),
            f"Counter mismatch at episode {index + 1}",
        )
        require(abs(float(percent) - wins / completed * 100) <= 0.050001, "Rounded cumulative SR")
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
            require(len(task_rates) == len(total_rates) == 1, "Missing or duplicated task summaries")
            require(
                abs(float(task_rates[0]) - sum(row["success"] for row in rows[-trials:]) / trials) < 1e-12,
                "Incorrect task summary",
            )
            require(abs(float(total_rates[0]) - wins / completed) < 1e-12, "Incorrect cumulative summary")
        else:
            require(not task_rates and not total_rates, "Premature task summary")
    require(int(one(r"Total episodes: (\d+)\b", text, "Final count")) == expected, "Final count mismatch")
    require(int(one(r"Total successes: (\d+)\b", text, "Final wins")) == wins, "Final wins mismatch")
    rate, percent = one(r"Overall success rate: ([\d.]+) \(([\d.]+)%\)", text, "Final SR")
    require(abs(float(rate) - wins / expected) <= 0.000050001, "Rounded final SR mismatch")
    require(abs(float(percent) - wins / expected * 100) <= 0.050001, "Rounded final percent mismatch")
    return rows
