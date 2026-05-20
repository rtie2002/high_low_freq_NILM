import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
HF_DIR = PROJECT_ROOT / "dataset_preprocess" / "high_frequency_data_extract"

sys.path.insert(0, str(HF_DIR))

from hf_feature import ADC_SCALE, compute_hf_features  # noqa: E402
from high_frequency_data_extract import decode_unix_time, get_calibration  # noqa: E402


def _load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _window_start_time(flac_path, window_index, win_sec, timezone):
    basename = os.path.basename(flac_path)
    try:
        start_unix = int(basename.split("-")[1].split("_")[0])
    except (IndexError, ValueError):
        start_unix = 0
    current_unix = start_unix + window_index * win_sec
    return current_unix, decode_unix_time(current_unix, timezone)


def _calibrate_block(block, config, v_step, i_step):
    v_idx = config["hyperparameters"]["channel_config"]["voltage_idx"]
    i_idx = config["hyperparameters"]["channel_config"]["current_idx"]
    v_t = block[:, v_idx] * ADC_SCALE * v_step
    i_t = block[:, i_idx] * ADC_SCALE * i_step
    return v_t, i_t


def _rms_like_spectrum(v_t, i_t, fs):
    n = len(i_t)
    window = np.hanning(n)
    window_norm = window / (window.sum() / n)

    v_fft = np.fft.rfft(v_t * window_norm)
    i_fft = np.fft.rfft(i_t * window_norm)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    v_amp = (np.abs(v_fft) * (2.0 / n)) / np.sqrt(2)
    i_amp = (np.abs(i_fft) * (2.0 / n)) / np.sqrt(2)
    return freqs, v_amp, i_amp


def _plot_hf_window(flac_path, config, window_index, output_path, show, waveform_display_sec):
    info = sf.info(flac_path)
    actual_sr = info.samplerate
    win_sec = config["hyperparameters"]["window_size_seconds"]
    chunk_size = int(actual_sr * win_sec)
    sample_start = window_index * chunk_size

    block, _ = sf.read(
        flac_path,
        start=sample_start,
        frames=chunk_size,
        always_2d=True,
    )

    if len(block) == 0:
        raise ValueError(f"Window {window_index} starts beyond the end of the file.")

    v_step, i_step, cal_file, house_id = get_calibration(
        flac_path,
        config["hyperparameters"].get("house_id"),
    )
    v_t, i_t = _calibrate_block(block, config, v_step, i_step)
    feat = compute_hf_features(block, config, v_step, i_step)

    current_unix, readable_time = _window_start_time(
        flac_path,
        window_index,
        win_sec,
        config["hyperparameters"].get("timezone", "Europe/London"),
    )

    fs = actual_sr
    freqs, _v_amp, i_amp = _rms_like_spectrum(v_t, i_t, fs)
    f0 = config["hyperparameters"]["high_frequency"]["mains_frequency"]
    harmonic_orders = config["features_to_extract"]["harmonic_analysis"]["orders"]

    max_waveform_samples = max(2, min(len(block), int(waveform_display_sec * fs)))
    t_wave = np.arange(max_waveform_samples) / fs

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"HF Feature Window {window_index} | {readable_time} | "
        f"{win_sec}s, {len(block)} samples | House {house_id}",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(t_wave, v_t[:max_waveform_samples], color="#d62728", lw=1.0, label="Voltage (V)")
    ax2 = ax.twinx()
    ax2.plot(t_wave, i_t[:max_waveform_samples], color="#1f77b4", lw=1.0, label="Current (A)")
    ax.set_title(f"Calibrated waveform preview ({waveform_display_sec:.3f}s shown)")
    ax.set_xlabel("Seconds from window start")
    ax.set_ylabel("Voltage (V)", color="#d62728")
    ax2.set_ylabel("Current (A)", color="#1f77b4")
    ax.grid(alpha=0.25)
    ax.text(
        0.01,
        0.96,
        f"V_rms={feat.get('V_rms', np.nan):.2f} V\n"
        f"I_rms={feat.get('I_rms', np.nan):.4f} A\n"
        f"P={feat.get('P_active', np.nan):.2f} W\n"
        f"PF={feat.get('PF', np.nan):.3f}\n"
        f"Fci={feat.get('Fci', np.nan):.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    ax = axes[0, 1]
    freq_mask = freqs <= 1000
    ax.plot(freqs[freq_mask], i_amp[freq_mask], color="#222222", lw=1.0)
    for order in harmonic_orders:
        harmonic_freq = order * f0
        if harmonic_freq <= 1000:
            ax.axvline(harmonic_freq, color="#ff7f0e", alpha=0.5, lw=0.8)
            ax.text(harmonic_freq, ax.get_ylim()[1] * 0.88, f"I{order}", rotation=90, ha="right")
    ax.set_title("Current RMS-like FFT spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Current spectral amplitude (A RMS-like)")
    ax.grid(alpha=0.25)
    ax.text(
        0.01,
        0.96,
        f"I1={feat.get('I1', np.nan):.4f}\n"
        f"I3={feat.get('I3', np.nan):.4f}\n"
        f"I5={feat.get('I5', np.nan):.4f}\n"
        f"THDI={feat.get('THDI', np.nan):.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    ax = axes[1, 0]
    env_keys = [f"I_env_{i}" for i in range(8) if f"I_env_{i}" in feat]
    env_vals = [feat[k] for k in env_keys]
    ax.bar(env_keys, env_vals, color="#2ca02c")
    ax.set_title("Normalized current spectral envelope")
    ax.set_ylabel("Normalized log energy")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    dwt_keys = [f"DWT_E{i}" for i in range(5) if f"DWT_E{i}" in feat]
    dwt_vals = [feat[k] for k in dwt_keys]
    ax.bar(dwt_keys, dwt_vals, color="#9467bd")
    ax.set_title("DWT sub-band mean squared energy")
    ax.set_ylabel("Mean squared coefficient energy")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.96,
        "DWT_E0: cA4, approx. 0-500 Hz\n"
        "DWT_E1: cD4, approx. 500-1000 Hz\n"
        "DWT_E2: cD3, approx. 1000-2000 Hz\n"
        "DWT_E3: cD2, approx. 2000-4000 Hz\n"
        "DWT_E4: cD1, approx. 4000-8000 Hz",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)

    print(f"[OK] Saved: {output_path}")
    print(f"[INFO] Calibration: {cal_file}")
    print(f"[INFO] Window start Unix: {current_unix}")
    print(f"[INFO] Window readable_time: {readable_time}")
    print(f"[INFO] Feature preview: V_rms={feat.get('V_rms'):.3f}, "
          f"I_rms={feat.get('I_rms'):.5f}, P_active={feat.get('P_active'):.3f}, "
          f"PF={feat.get('PF'):.4f}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the exact 6-second HF feature window used by hf_feature.py."
    )
    parser.add_argument("--flac", required=True, help="Path to a UK-DALE high-frequency .flac file.")
    parser.add_argument(
        "--config",
        default=str(HF_DIR / "hf_config.yaml"),
        help="Path to hf_config.yaml.",
    )
    parser.add_argument("--window-index", type=int, default=0, help="6-second window index to visualize.")
    parser.add_argument(
        "--waveform-display-sec",
        type=float,
        default=0.08,
        help="Seconds of waveform to show inside the selected 6-second feature window.",
    )
    parser.add_argument("--output", default=None, help="PNG output path. Defaults to data_quality_checking/visualizations.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively after saving.")
    args = parser.parse_args()

    flac_path = Path(args.flac).resolve()
    config = _load_config(args.config)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_dir = SCRIPT_DIR / "visualizations"
        output_path = output_dir / f"hf_window_{flac_path.stem}_w{args.window_index:04d}.png"

    _plot_hf_window(
        str(flac_path),
        config,
        args.window_index,
        output_path,
        args.show,
        args.waveform_display_sec,
    )


if __name__ == "__main__":
    main()
