import numpy as np
from scipy import stats as scipy_stats
try:
    import pywt
except ImportError:
    pywt = None

"""
NILM High-Frequency Feature Engine  ―  Multi-View Signal Representation
------------------------------------------------------------------------
Design Philosophy (DL-First):
    This module is a *signal detector*, not a *physics simulator*.

    LEVEL 1 — Raw Statistics   : robust, noise-tolerant, no FFT assumptions
    LEVEL 2 — Light Spectral   : weak FFT, leakage is OK, let the model learn
    LEVEL 3 — Wavelet (TF)     : transient signatures, the soul of NILM

    ✅ Kept  : RMS, Crest Factor, waveform shape stats, band power, DWT energy
    ❌ Removed: S²=P²+Q²+D² closure, strict harmonic summation identity,
                cycle-perfect FFT, complex power averaging, Budeanu decomposition
    Rationale: Real-world NILM data (UK-DALE) has grid drift, async sampling,
               and appliance interference. Hard physical constraints become
               brittle noise in this environment. DL models benefit more from
               redundant, multi-view signal representations.
"""

ADC_SCALE = 2**31


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1 — RAW WAVEFORM STATISTICS
# Features: V_rms, I_rms, P_active, Q_reactive, S_apparent, PF, Fcv, Fci
#           + Skewness, Kurtosis, ZCR (added for DL robustness)
# No FFT. No cycle alignment. Pure time-domain.
# ─────────────────────────────────────────────────────────────────────────────
def get_time_domain_features(v_t, i_t, config):
    """
    Level 1: Raw waveform statistics.

    All calculations are purely time-domain — no FFT, no cycle alignment.
    This makes features maximally robust to grid frequency drift and
    sampling jitter, which are common in real-world NILM datasets.
    """
    res = {}
    feat_cfg = config['features_to_extract']
    t_cfg = feat_cfg['time_domain']
    if not t_cfg['enabled']:
        return res

    N = len(v_t)
    if N < 2:
        return res

    # ── Core Power Statistics ────────────────────────────────────────────────
    v_rms = float(np.sqrt(np.mean(v_t ** 2)))
    i_rms = float(np.sqrt(np.mean(i_t ** 2)))

    # Active Power: time-domain expectation of instantaneous power
    p_active = float(np.mean(v_t * i_t))

    # Apparent Power: product of RMS values
    s_apparent = float(v_rms * i_rms)

    # Power Factor
    pf = float(p_active / (s_apparent + 1e-9))

    # Crest Factors
    v_peak = float(np.max(np.abs(v_t)))
    i_peak = float(np.max(np.abs(i_t)))
    fcv = float(v_peak / (v_rms + 1e-9))
    fci = float(i_peak / (i_rms + 1e-9))

    if t_cfg['v_rms']:     res['V_rms']      = v_rms
    if t_cfg['i_rms']:     res['I_rms']      = i_rms
    if t_cfg['p_active']:  res['P_active']   = p_active
    if t_cfg['s_apparent']: res['S_apparent'] = s_apparent
    if t_cfg['pf']:        res['PF']         = pf
    if t_cfg['fcv']:       res['Fcv']        = fcv
    if t_cfg['fci']:       res['Fci']        = fci

    # ── Shape Statistics (Now controllable via config) ────────────────────────
    s_cfg = feat_cfg.get('shape_statistics', {'enabled': False})
    if s_cfg['enabled']:
        if s_cfg.get('i_skew', True): res['I_skew']  = float(scipy_stats.skew(i_t))
        if s_cfg.get('i_kurt', True): res['I_kurt']  = float(scipy_stats.kurtosis(i_t))
        if s_cfg.get('v_skew', True): res['V_skew']  = float(scipy_stats.skew(v_t))
        if s_cfg.get('i_std', True):  res['I_std']   = float(np.std(i_t))
        if s_cfg.get('v_std', True):  res['V_std']   = float(np.std(v_t))

    return res


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — LIGHT SPECTRAL FEATURES
# Features: per-harmonic I_k, V_k magnitudes (no power closure)
#           + band power, spectral centroid, spectral entropy
# Philosophy: Give the DL model the raw spectral information.
#             Let the Attention mechanism decide what matters.
# ─────────────────────────────────────────────────────────────────────────────
def get_harmonic_analysis(v_t, i_t, config):
    """
    Level 2: Light spectral features via rfft magnitude spectrum.

    Key Changes from Physics approach:
    - NO complex power S_k = V_k × I_k*  (removed cross-domain coupling)
    - NO harmonic summation closure (P_total = P1 + PH etc.)
    - YES: raw magnitude of each harmonic bin → let DL learn relationships
    - YES: band power, spectral centroid, spectral entropy as global descriptors

    Leakage from grid frequency drift is intentionally allowed —
    it may itself carry discriminative information for load identification.
    """
    feat_cfg = config['features_to_extract']
    h_cfg = feat_cfg['harmonic_analysis']
    d_cfg = feat_cfg['distortion_metrics']

    hf_cfg = config['hyperparameters']['high_frequency']
    fs = hf_cfg['sampling_rate']
    M = int(fs / hf_cfg['mains_frequency'])  # samples per cycle = 320

    f = {}
    N = len(i_t)
    # Require at least 6 cycles for adequate frequency resolution (~8.3 Hz/bin)
    # 1-cycle (320 pts) gives only 50Hz resolution — too coarse to separate harmonics
    min_samples = 6 * M
    if N < min_samples:
        return f

    # ── Windowed FFT (no cycle alignment, use full block) ────────────────────
    # Apply Hann window to reduce spectral leakage at block boundaries.
    # We intentionally do NOT enforce cycle-perfect alignment here.
    window = np.hanning(N)
    # Normalize window to preserve amplitude
    window_norm = window / (window.sum() / N)

    v_fft = np.fft.rfft(v_t * window_norm)
    i_fft = np.fft.rfft(i_t * window_norm)

    freqs = np.fft.rfftfreq(N, d=1.0/fs)   # frequency axis in Hz

    # Amplitude spectra: Normalize to RMS values (peak / sqrt(2))
    # Why: sum(amp_rms**2) will now roughly equal I_rms^2 (Parseval-like identity)
    v_amp = (np.abs(v_fft) * (2.0 / N)) / np.sqrt(2)
    i_amp = (np.abs(i_fft) * (2.0 / N)) / np.sqrt(2)

    # ── Per-Harmonic Magnitudes ───────────────────────────────────────────────
    f0 = hf_cfg['mains_frequency']  # 50 Hz
    i1_amp = None
    v1_amp = None
    ih_sq = 0.0
    vh_sq = 0.0

    for k in h_cfg['orders']:
        target_freq = k * f0
        bin_idx = int(round(target_freq * N / fs))
        if bin_idx >= len(i_amp): break

        # Band Energy extraction: Hz-based window (+/- 15Hz)
        hz_bw = h_cfg.get('harmonic_band_hz', 15.0)
        bin_bw = max(1, int(round(hz_bw / (fs / N))))
        s_idx = max(0, bin_idx - bin_bw)
        e_idx = min(len(i_amp), bin_idx + bin_bw + 1)
        
        # Now amplitude is already RMS
        ik = float(np.sqrt(np.sum(i_amp[s_idx:e_idx]**2)))
        vk = float(np.sqrt(np.sum(v_amp[s_idx:e_idx]**2)))

        if h_cfg['enabled']:
            if h_cfg['ik']: f[f'I{k}'] = ik
            if h_cfg['vk']: f[f'V{k}'] = vk

        if k == 1:
            i1_amp, v1_amp = ik, vk
        if k >= 2:
            ih_sq += ik ** 2
            vh_sq += vk ** 2

    # ── Distortion Metrics ───────────────────────────────────────────────────
    if d_cfg['enabled'] and i1_amp is not None:
        ih_h, vh_h = float(np.sqrt(ih_sq)), float(np.sqrt(vh_sq))
        if d_cfg['ih']:   f['IH']   = ih_h
        if d_cfg['vh']:   f['VH']   = vh_h
        if d_cfg['thdi']: f['THDI'] = float(ih_h / (i1_amp + 1e-9))
        if d_cfg['thdv']: f['THDV'] = float(vh_h / (v1_amp + 1e-9))

    # ── Band Power (normalized to match physical power level) ─────────────────
    b_cfg = feat_cfg.get('band_power', {'enabled': False})
    if b_cfg['enabled']:
        def _band_power(amp_spectrum, f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            return float(np.sum(amp_spectrum[mask] ** 2))

        if b_cfg.get('i_bp_low', True):  f['I_BP_low']  = _band_power(i_amp, 50,   500)
        if b_cfg.get('i_bp_mid', True):  f['I_BP_mid']  = _band_power(i_amp, 500,  2000)
        if b_cfg.get('i_bp_high', True): f['I_BP_high'] = _band_power(i_amp, 2000, 8000)
        if b_cfg.get('v_bp_low', True):  f['V_BP_low']  = _band_power(v_amp, 50,   500)

    # ── Spectral Descriptors ──────────────────────────────────────────────────
    sd_cfg = feat_cfg.get('spectral_descriptors', {'enabled': False})
    if sd_cfg['enabled']:
        if sd_cfg.get('i_spec_entropy', True):
            entropy_mask = freqs <= 3000
            i_amp_sq_ent = i_amp[entropy_mask] ** 2
            total_power_ent = i_amp_sq_ent.sum() + 1e-12
            prob = i_amp_sq_ent / total_power_ent
            prob = prob[prob > 0]
            spec_entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
            f['I_spec_entropy'] = spec_entropy

    # ── Spectral Envelope (Log-Frequency Shape) ─────────────────────────────
    e_cfg = feat_cfg.get('spectral_envelope', {'enabled': False})
    if e_cfg['enabled']:
        log_bands = [
            (0, 100),       # DC + fundamental neighborhood
            (100, 200),     # 2nd-4th harmonic zone
            (200, 400),     # 4th-8th harmonic zone
            (400, 800),     # 8th-16th harmonic zone
            (800, 1600),    # motor commutation / arc zone
            (1600, 3200),   # SMPS switching frequency zone
            (3200, 6400),   # high-freq transient zone
            (6400, 8000),   # ultra-high-freq residual
        ]
        for i, (f_lo, f_hi) in enumerate(log_bands):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            band_amp = i_amp[mask]
            if len(band_amp) > 0:
                energy = float(np.sum(band_amp ** 2))
                f[f'I_env_{i}'] = float(np.log1p(energy))
            else:
                f[f'I_env_{i}'] = 0.0

        # Normalize envelope to capture spectral SHAPE, not magnitude
        env_keys = [k for k in f if k.startswith('I_env_')]
        env_vals = np.array([f[k] for k in env_keys])
        env_total = env_vals.sum()
        if env_total > 1e-12:
            for k in env_keys:
                f[k] = float(f[k] / env_total)

    return f


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3 — WAVELET TIME-FREQUENCY FEATURES
# Features: DWT sub-band energy per level
# This is the most important layer for NILM — transient signatures are
# unique fingerprints of each appliance's switching behaviour.
# ─────────────────────────────────────────────────────────────────────────────
def get_wavelet_features(i_t, feat_cfg):
    """
    Level 3: Discrete Wavelet Transform (DWT) sub-band energy.

    Frequency bands at 16kHz sampling (db4, 4 levels):
        E0 (approx): ~0–500 Hz   (fundamental + low harmonics)
        E1 (D1):     4–8 kHz     (SMPS switching noise)
        E2 (D2):     2–4 kHz     (arcing, fast transients)
        E3 (D3):     1–2 kHz     (motor commutation)
        E4 (D4):     500Hz–1kHz  (slow transients, startup)

    Uses Mean Squared Energy (MSE) normalisation so that energy is
    invariant to window length — critical for consistent comparison
    across different block sizes.
    """
    f = {}
    cfg = feat_cfg['wavelet_domain']
    if pywt is None or not cfg['enabled']:
        return f

    coeffs = pywt.wavedec(i_t, 'db4', level=cfg['levels'])
    for i, c in enumerate(coeffs):
        f[f'DWT_E{i}'] = float(np.mean(np.square(c)))

    return f


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
def compute_hf_features(block, config, v_step, i_step):
    """
    Multi-domain feature orchestrator.

    Feature count breakdown (approximate, depends on config):
        Level 1 — Raw Statistics:  ~14 features
        Level 2 — Light Spectral:  ~20–28 features (harmonic bins + band power + descriptors)
        Level 3 — Wavelet:         ~5  features
        Total:                     ~40–47 features

    All features are DL-friendly:
        ✅ Robust to grid frequency drift (no strict cycle-lock)
        ✅ Redundant (some overlap is intentional — helps gradient flow)
        ✅ Low assumption (no Budeanu closure, no complex power identity)
    """
    v_idx = config['hyperparameters']['channel_config']['voltage_idx']
    i_idx = config['hyperparameters']['channel_config']['current_idx']
    v_t = block[:, v_idx] * ADC_SCALE * v_step
    i_t = block[:, i_idx] * ADC_SCALE * i_step

    res = {}
    res.update(get_time_domain_features(v_t, i_t, config))
    res.update(get_harmonic_analysis(v_t, i_t, config))
    res.update(get_wavelet_features(i_t, config['features_to_extract']))

    return res
