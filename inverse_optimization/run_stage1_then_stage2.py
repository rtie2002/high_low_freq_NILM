"""Run Stage 1 training, then run Stage 2 using that new Stage-1 checkpoint."""

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
STAGE1_SCRIPT = PROJECT_DIR / "stage1_forward" / "forward_model_train.py"
STAGE2_SCRIPT = PROJECT_DIR / "stage2_inverse_optimization" / "inverse_optimization.py"


def saved_checkpoints():
    """Return every existing Stage-1 best checkpoint under results/."""
    return {path.resolve() for path in RESULTS_DIR.glob("*/best.pt")}


def main():
    # Remember old checkpoints so an earlier experiment can never be selected.
    previous_checkpoints = saved_checkpoints()

    print("\n========== STAGE 1: FORWARD MODEL TRAINING ==========")
    subprocess.run([sys.executable, str(STAGE1_SCRIPT)], cwd=PROJECT_DIR, check=True)

    new_checkpoints = saved_checkpoints() - previous_checkpoints
    if not new_checkpoints:
        raise FileNotFoundError("Stage 1 finished without creating a new results/.../best.pt checkpoint.")

    # Stage 1 creates a timestamped directory, so this is normally one path.
    checkpoint_path = max(new_checkpoints, key=lambda path: path.stat().st_mtime)
    print(f"\nNew Stage-1 checkpoint: {checkpoint_path}")

    print("\n========== STAGE 2: INVERSE OPTIMISATION ==========")
    subprocess.run(
        [sys.executable, str(STAGE2_SCRIPT), "--checkpoint", str(checkpoint_path)],
        cwd=PROJECT_DIR,
        check=True,
    )


if __name__ == "__main__":
    main()
