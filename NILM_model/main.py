import argparse
from pathlib import Path
import sys


BASELINE_DIR = Path(__file__).resolve().parent
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Universal NILM baseline entry point.",
        add_help=True,
    )
    parser.add_argument("--model", choices=["sgn"], required=True)
    parser.add_argument("--mode", choices=["train", "inference", "train_inference"], required=True)
    args, model_args = parser.parse_known_args()
    return args, model_args


def _replace_or_add(args: list[str], name: str, value: str) -> list[str]:
    args = list(args)
    if name in args:
        idx = args.index(name)
        if idx + 1 < len(args):
            args[idx + 1] = value
            return args
    return args + [name, value]


def _checkpoint_appliance(checkpoint: Path) -> str:
    name = checkpoint.stem
    return name.removeprefix("best_")


def run_sgn(mode: str, model_args: list[str]) -> None:
    from models import sgn_pipeline

    if mode == "train":
        sgn_pipeline.train_main(model_args)
        return

    if mode == "inference":
        sgn_pipeline.inference_main(model_args)
        return

    checkpoints = sgn_pipeline.train_main(model_args)
    for checkpoint in checkpoints:
        inference_args = _replace_or_add(model_args, "--checkpoint", str(checkpoint))
        inference_args = _replace_or_add(inference_args, "--appliance", _checkpoint_appliance(checkpoint))
        sgn_pipeline.inference_main(inference_args, allow_unknown=True)


def main() -> None:
    args, model_args = parse_args()
    if args.model == "sgn":
        run_sgn(args.mode, model_args)
        return
    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
