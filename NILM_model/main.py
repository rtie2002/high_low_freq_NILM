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


def _format_metrics_summary(results: list[dict]) -> str:
    rows = []
    for result in results:
        per_appliance = result.get("per_appliance", {})
        if per_appliance:
            for appliance, metrics in per_appliance.items():
                rows.append(
                    (
                        appliance,
                        float(metrics["mae"]),
                        float(metrics["sae"]),
                        float(metrics["f1"]),
                    )
                )
        else:
            average = result.get("average", {})
            rows.append(
                (
                    "average",
                    float(average["mae"]),
                    float(average["sae"]),
                    float(average["f1"]),
                )
            )

    if not rows:
        return "\n== Final Test Summary ==\nNo metrics were returned."

    avg_mae = sum(row[1] for row in rows) / len(rows)
    avg_sae = sum(row[2] for row in rows) / len(rows)
    avg_f1 = sum(row[3] for row in rows) / len(rows)

    name_width = max(15, max(len(row[0]) for row in rows))
    lines = [
        "\n== Final Test Summary ==",
        f"{'appliance':<{name_width}}  {'MAE':>10}  {'SAE':>10}  {'F1':>10}",
        f"{'-' * name_width}  {'-' * 10}  {'-' * 10}  {'-' * 10}",
    ]
    for appliance, mae, sae, f1 in rows:
        lines.append(f"{appliance:<{name_width}}  {mae:10.3f}  {sae:10.3f}  {f1:10.3f}")
    if len(rows) > 1:
        lines.extend(
            [
                f"{'-' * name_width}  {'-' * 10}  {'-' * 10}  {'-' * 10}",
                f"{'average':<{name_width}}  {avg_mae:10.3f}  {avg_sae:10.3f}  {avg_f1:10.3f}",
            ]
        )
    return "\n".join(lines)


def run_sgn(mode: str, model_args: list[str]) -> None:
    from models import sgn_pipeline

    if mode == "train":
        sgn_pipeline.train_main(model_args)
        return

    if mode == "inference":
        result = sgn_pipeline.inference_main(model_args)
        print(_format_metrics_summary([result]))
        return

    checkpoints = sgn_pipeline.train_main(model_args)
    results = []
    for checkpoint in checkpoints:
        inference_args = _replace_or_add(model_args, "--checkpoint", str(checkpoint))
        results.append(sgn_pipeline.inference_main(inference_args, allow_unknown=True))
    print(_format_metrics_summary(results))


def main() -> None:
    args, model_args = parse_args()
    if args.model == "sgn":
        run_sgn(args.mode, model_args)
        return
    raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
