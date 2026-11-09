"""Report frozen predictions by true concurrent target count; no tuning."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def audit(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        truth = data["y_true_on"].astype(bool)
        prediction = data["y_pred_on"].astype(bool)
        target = data["y_true_watts"]
        power = data["y_pred_watts"]
        if truth.shape != prediction.shape or target.shape != power.shape or truth.shape != target.shape:
            raise ValueError(f"Incompatible prediction arrays: {path}")
        counts = truth.sum(axis=1)
        rows = []
        for label, mask in [("0", counts == 0), ("1", counts == 1), ("2", counts == 2), ("3", counts == 3), ("4-5", counts >= 4)]:
            n = int(mask.sum())
            if not n:
                rows.append({"source": str(path), "concurrent": label, "samples": 0})
                continue
            t, p = truth[mask], prediction[mask]
            tp = (t & p).sum(axis=0)
            fp = (~t & p).sum(axis=0)
            fn = (t & ~p).sum(axis=0)
            denom = 2 * tp + fp + fn
            # Absent classes are undefined, not perfect or zero. Report support.
            f1 = np.divide(2 * tp, denom, out=np.full(len(tp), np.nan), where=denom > 0)
            errors = np.abs(target[mask] - power[mask])
            on_count = t.sum(axis=0)
            on_mae = np.divide((errors * t).sum(axis=0), on_count, out=np.full(len(tp), np.nan), where=on_count > 0)
            rows.append({"source": str(path), "concurrent": label, "samples": n,
                         "exact_match": float((t == p).all(axis=1).mean()),
                         "macro_f1": float(np.nanmean(f1)) if np.isfinite(f1).any() else np.nan,
                         "f1_classes": int(np.isfinite(f1).sum()),
                         "on_mae": float(np.nanmean(on_mae)) if np.isfinite(on_mae).any() else np.nan,
                         "on_mae_classes": int(np.isfinite(on_mae).sum()),
                         "mae": float(errors.mean())})
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for path in args.predictions for row in audit(path)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
