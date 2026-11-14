"""Resume one saved exploration-list run until completion or a non-retryable error."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import sleep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    delay = 10
    while True:
        run = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if run["completed"]:
            return
        config = run["tree_config"]
        before = len(run.get("interrupted_case", {}).get("records", []))
        result = subprocess.run([
            sys.executable, "-u", "-m", "experiments.test_hard_quick",
            "--method", "mcts-tree", "--tree-policy", config["policy"],
            "--child-limit", str(config["child_limit"]),
            "--duplicate-reward", str(config["duplicate_reward"]),
            "--exploration-list", config["exploration_list"], "--provider-token-limit",
            "--pause-file", "results/PAUSE_EXPLORATION_LIST_V2",
            "--resume", str(args.checkpoint),
        ])
        updated = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if updated["completed"]:
            return
        records = updated.get("interrupted_case", {}).get("records", [])
        last_error = records[-1].get("error", "") if records else ""
        if (len(updated["cases"]) == len(run["cases"]) and len(records) <= before
                or not (last_error == "Model connection failed or timed out"
                        or "HTTP 429" in last_error or "HTTP 5" in last_error)):
            raise SystemExit(result.returncode or 1)
        print(f"Transient error. Resuming saved progress in {delay}s.", flush=True)
        sleep(delay)
        delay = min(delay * 2, 120)


if __name__ == "__main__":
    main()
