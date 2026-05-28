import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEAP_PYTHON_DIR

EXPECTED_S01_FIRST_LABEL = np.array([7.71, 7.60, 6.90, 7.83])
TOL = 1e-2


def load_dat(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def main():
    files = sorted(DEAP_PYTHON_DIR.glob("s*.dat"))

    if len(files) != 32:
        raise RuntimeError(f"Expected 32 subject files, found {len(files)} in {DEAP_PYTHON_DIR}")

    labels_all = []

    for path in files:
        d = load_dat(path)

        if d["data"].shape != (40, 40, 8064):
            raise RuntimeError(f"{path.name} bad data shape: {d['data'].shape}")

        if d["labels"].shape != (40, 4):
            raise RuntimeError(f"{path.name} bad label shape: {d['labels'].shape}")

        if np.min(d["labels"]) < 1.0 or np.max(d["labels"]) > 9.0:
            raise RuntimeError(
                f"{path.name} has labels outside 1-9 range: "
                f"min={np.min(d['labels'])}, max={np.max(d['labels'])}"
            )

        labels_all.append(d["labels"])

    s01 = load_dat(DEAP_PYTHON_DIR / "s01.dat")
    s01_first = s01["labels"][0]

    if not np.allclose(s01_first, EXPECTED_S01_FIRST_LABEL, atol=TOL):
        raise RuntimeError(
            "s01 first label does not match expected DEAP label.\n"
            f"Expected: {EXPECTED_S01_FIRST_LABEL}\n"
            f"Got:      {s01_first}"
        )

    labels = np.concatenate(labels_all, axis=0)

    print("DEAP integrity check passed.")
    print(f"Files: {len(files)}")
    print(f"Labels shape: {labels.shape}")
    print("Global mins:", labels.min(axis=0))
    print("Global maxs:", labels.max(axis=0))
    print("Global means:", labels.mean(axis=0))


if __name__ == "__main__":
    main()