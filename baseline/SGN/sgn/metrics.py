import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def sae(y_true: np.ndarray, y_pred: np.ndarray, period: int = 1200) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    n_periods = len(y_true) // period
    if n_periods == 0:
        return float("nan")
    total_error = 0.0
    for idx in range(n_periods):
        start = idx * period
        end = start + period
        total_error += abs(float(np.sum(y_true[start:end]) - np.sum(y_pred[start:end])))
    return float(total_error / (period * n_periods))


def f1_score_binary(y_true_on: np.ndarray, y_pred_on: np.ndarray) -> float:
    y_true_on = np.asarray(y_true_on).astype(bool).reshape(-1)
    y_pred_on = np.asarray(y_pred_on).astype(bool).reshape(-1)
    tp = np.logical_and(y_true_on, y_pred_on).sum()
    fp = np.logical_and(~y_true_on, y_pred_on).sum()
    fn = np.logical_and(y_true_on, ~y_pred_on).sum()
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float((2 * tp) / denom)


def compute_metrics(
    y_true_watts: np.ndarray,
    y_pred_watts: np.ndarray,
    y_true_on: np.ndarray,
    y_pred_on_prob: np.ndarray,
    sae_period: int = 1200,
) -> dict[str, float]:
    y_pred_watts = np.maximum(y_pred_watts, 0.0)
    y_pred_on = y_pred_on_prob >= 0.5
    return {
        "mae": mae(y_true_watts, y_pred_watts),
        "sae": sae(y_true_watts, y_pred_watts, sae_period),
        "f1": f1_score_binary(y_true_on, y_pred_on),
    }

