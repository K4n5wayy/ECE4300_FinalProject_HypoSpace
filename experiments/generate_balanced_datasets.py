"""Build the reduced comparison datasets from fixed HypoSpace pools."""
import json
from pathlib import Path
import subprocess
import sys

from .direct_evaluation import ROOT, UPSTREAM

HARD = ROOT / "datasets/hard"
BALANCED = ROOT / "datasets/balanced"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, data):
    (BALANCED / f"{name}.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


def main():
    BALANCED.mkdir(exist_ok=True)
    causal = read(HARD / "causal.json")
    rows = causal["datasets"]
    causal["datasets"] = [rows[i - 1] for i in (1, 10, 16, 17, 20, 21, 22, 29)]
    causal["metadata"]["n_datasets"] = len(causal["datasets"])
    write("causal", causal)

    voxel = read(HARD / "3d.json")
    for blocks, limit in ((2, 4), (4, 8)):
        path = BALANCED / f"voxel_{blocks}_pool.json"
        subprocess.run([
            sys.executable, str(UPSTREAM / "3d/generate_3d_dataset_complete.py"),
            "--max-blocks", str(blocks), "--fixed", "--seed", "33550336", "--output", str(path),
        ], check=True)
        selected = read(path)["observation_sets"]
        rows = [selected[i * len(selected) // limit] for i in range(limit)]
        if blocks == 2:
            voxel["observation_sets"] = voxel["observation_sets"][:19] + rows
        else:
            voxel["observation_sets"].extend(rows)
        path.unlink()
    voxel["metadata"].update(n_observation_sets=len(voxel["observation_sets"]), max_blocks=4)
    write("3d", voxel)

    full = read(HARD / "boolean_pool.json")
    write("boolean", full)
    extended = read(BALANCED / "boolean_extended_pool.json")
    options = {0: []}
    for row in sum(extended["datasets_by_n_observations"].values(), []):
        size = row["n_compatible_expressions"]
        for total, chosen in list(options.items()):
            if total + size <= 225 and (total + size not in options
                                        or len(chosen) + 1 < len(options[total + size])):
                options[total + size] = [*chosen, row]
    groups = {}
    for row in options[max(options)]:
        row["observation_set_id"] = "extended_" + row["observation_set_id"]
        groups.setdefault(str(row["n_observations"]), []).append(row)
    extended["datasets_by_n_observations"] = groups
    extended["metadata"]["total_observation_sets"] = sum(map(len, groups.values()))
    write("boolean_extended", extended)


if __name__ == "__main__":
    main()
