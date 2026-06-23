from pathlib import Path
import sys


NILM_MODEL_DIR = Path(__file__).resolve().parents[2]
if str(NILM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NILM_MODEL_DIR))

from models.sgn_pipeline import train_main


def main(argv: list[str] | None = None):
    return train_main(argv)


if __name__ == "__main__":
    main()
