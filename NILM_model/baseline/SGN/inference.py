from pathlib import Path
import sys


NILM_MODEL_DIR = Path(__file__).resolve().parents[2]
if str(NILM_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NILM_MODEL_DIR))

from models.sgn_pipeline import inference_main


def main(argv: list[str] | None = None, *, allow_unknown: bool = False):
    return inference_main(argv, allow_unknown=allow_unknown)


if __name__ == "__main__":
    main()
