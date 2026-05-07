import numpy as np
try:
    import pywt
except ImportError:
    pywt = None

"""
NILM High-Frequency Feature Engine (Research Grade - Cycle Averaged)
-------------------------------------------------------------------
This version implements cycle-by-cycle calculation and averaging to 
ensure 1:1 mathematical parity with the ELECTRIMACS 2017 framework.
"""

ADC_SCALE = 1.0 / (2**15) 

def get_time_domain_features(v_t, i_t, config):
    """
    Calculates time-domain features by averaging individual cycles.
    This ensures robustness against noise and start-phase alignment.
    """
    res = {}
    feat_cfg = config['features_to_extract']
    t_cfg = feat_cfg['time_domain']
    if not t_cfg['enabled']: return res
    
    # 1. Cycle Splitting Logic
    hf_cfg = config['hyperparameters']['high_frequency']
    samples_per_cycle = int(hf_cfg['sampling_rate'] / hf_cfg['mains_frequency'])
    num_cycles = len(v_t) // samples_per_cycle
    
    if num_cycles < 1: return res

    # Reshape into [num_cycles, samples_per_cycle]
    v_cycles = v_t[:num_cycles*samples_per_cycle].reshape(num_cycles, samples_per_cycle)
    i_cycles = i_t[:num_cycles*samples_per_cycle].reshape(num_cycles, samples_per_cycle)
    
    # 2. Per-Cycle Calculations (Vectorized for Speed)
    
    # RMS (Eq. 9)
    v_rms_cycles = np.sqrt(np.mean(np.square(v_cycles), axis=1))
    i_rms_cycles = np.sqrt(np.mean(np.square(i_cycles), axis=1))
    
    # Power Triangle (Eq. 14-16)
    p_active_cycles = np.mean(v_cycles * i_cycles, axis=1)
    s_apparent_cycles = v_rms_cycles * i_rms_cycles
    q_reactive_cycles = np.sqrt(np.maximum(0, s_apparent_cycles**2 - p_active_cycles**2))
    pf_cycles = p_active_cycles / (s_apparent_cycles + 1e-6)
    
    # Crest Factor (Eq. 20)
    v_peak_cycles = np.max(np.abs(v_cycles), axis=1)
    i_peak_cycles = np.max(np.abs(i_cycles), axis=1)
    cf_v_cycles = v_peak_cycles / (v_rms_cycles + 1e-6)
    cf_i_cycles = i_peak_cycles / (i_rms_cycles + 1e-6)

    # 3. Aggregation (Averaging over the 6s window)
    if t_cfg['v_rms']: res['V_rms'] = round(float(np.mean(v_rms_cycles)), 4)
    if t_cfg['i_rms']: res['I_rms'] = round(float(np.mean(i_rms_cycles)), 6)
    
    if t_cfg['p_active']: res['P_active'] = round(float(np.mean(p_active_cycles)), 2)
    if t_cfg['q_reactive']: res['Q_reactive'] = round(float(np.mean(q_reactive_cycles)), 2)
    if t_cfg['s_apparent']: res['S_apparent'] = round(float(np.mean(s_apparent_cycles)), 2)
    if t_cfg['pf']: res['PF'] = round(float(np.mean(pf_cycles)), 4)
    
    if t_cfg['fcv']: res['Fcv'] = round(float(np.mean(cf_v_cycles)), 4)
    if t_cfg['fci']: res['Fci'] = round(float(np.mean(cf_i_cycles)), 4)
    
    return res

def get_harmonic_analysis(v_t, i_t, config):
    """
    Exhaustive Atomic Harmonic Analysis with Cycle Averaging.
    Matches Eq. 1 - 19 of ELECTRIMACS 2017.
    """
    feat_cfg = config['features_to_extract']
    h_cfg = feat_cfg['harmonic_analysis']
    d_cfg = feat_cfg['distortion_metrics']
    
    hf_cfg = config['hyperparameters']['high_frequency']
    samples_per_cycle = int(hf_cfg['sampling_rate'] / hf_cfg['mains_frequency']) 
    num_cycles = len(v_t) // samples_per_cycle
    
    f = {}
    if num_cycles < 1: return f

    # 1. Cycle-Averaging FFT (Essential for noise reduction)
    v_cycles = v_t[:num_cycles*samples_per_cycle].reshape(num_cycles, samples_per_cycle)
    i_cycles = i_t[:num_cycles*samples_per_cycle].reshape(num_cycles, samples_per_cycle)
    v_fft = np.mean(np.fft.rfft(v_cycles, axis=1), axis=0)
    i_fft = np.mean(np.fft.rfft(i_cycles, axis=1), axis=0)
    
    scale = 2.0 / (samples_per_cycle**2)
    v_mags = np.abs(v_fft) * (2.0 / samples_per_cycle)
    i_mags = np.abs(i_fft) * (2.0 / samples_per_cycle)
    
    # Internal references (Fundamental k=1)
    S1_c = v_fft[1] * np.conj(i_fft[1]) * scale
    i1, v1, s1 = i_mags[1]/np.sqrt(2), v_mags[1]/np.sqrt(2), np.abs(S1_c)
    p1, q1 = np.real(S1_c), np.imag(S1_c)

    # Aggregates for distortion
    ph, qh, ih_sq, vh_sq = 0, 0, 0, 0
    
    # 2. Individual Harmonic Loop (k=1 to 15)
    for k in range(1, 16):
        if k >= len(v_fft): break
        Sk_c = v_fft[k] * np.conj(i_fft[k]) * scale
        pk, qk, sk = np.real(Sk_c), np.imag(Sk_c), np.abs(Sk_c)
        ik, vk = i_mags[k]/np.sqrt(2), v_mags[k]/np.sqrt(2)

        if h_cfg['enabled'] and k in h_cfg['orders']:
            if h_cfg['pk']: f[f'P{k}'] = pk
            if h_cfg['qk']: f[f'Q{k}'] = qk
            if h_cfg['ik']: f[f'I{k}'] = ik
            if h_cfg['sk']: f[f'S{k}'] = sk
            if h_cfg['vk']: f[f'V{k}'] = vk
        
        if k >= 2:
            ph += pk; qh += qk; ih_sq += ik**2; vh_sq += vk**2

    # 3. Distortion Metrics
    if d_cfg['enabled']:
        ih_h, vh_h = np.sqrt(ih_sq), np.sqrt(vh_sq)
        sh_h = vh_h * ih_h
        
        if d_cfg['ph']: f['PH'] = ph
        if d_cfg['qh']: f['QH'] = qh
        if d_cfg['ih']: f['IH'] = ih_h
        if d_cfg['vh']: f['VH'] = vh_h
        if d_cfg['sh']: f['SH'] = sh_h
        
        # Aggregate P, Q, S for D calculation (Eq. 14-16)
        p_t, q_t = p1 + ph, q1 + qh
        i_t, v_t = np.sqrt(i1**2 + ih_sq), np.sqrt(v1**2 + vh_sq)
        s_tot = v_t * i_t
        
        if d_cfg['d']: f['D'] = np.sqrt(max(0, s_tot**2 - p_t**2 - q_t**2))
        if d_cfg['thdi']: f['THDI'] = ih_h / (i1 + 1e-6)
        if d_cfg['thdv']: f['THDV'] = vh_h / (v1 + 1e-6)
        
        if d_cfg['di'] or d_cfg['dv'] or d_cfg['sn']:
            di, dv = s1 * (ih_h/(i1+1e-6)), s1 * (vh_h/(v1+1e-6))
            if d_cfg['di']: f['DI'] = di
            if d_cfg['dv']: f['DV'] = dv
            if d_cfg['sn']: f['SN'] = np.sqrt(di**2 + dv**2 + sh_h**2)

    return f

def get_wavelet_features(i_t, feat_cfg):
    """Extracts DWT energy levels (Transient)."""
    f = {}
    cfg = feat_cfg['wavelet_domain']
    if pywt is None or not cfg['enabled']: return f
    coeffs = pywt.wavedec(i_t, 'db4', level=cfg['levels'])
    for i, c in enumerate(coeffs):
        f[f'DWT_E{i}'] = round(float(np.sum(np.square(c))), 6)
    return f

def compute_hf_features(block, config, v_step, i_step):
    """Orchestrator for multi-domain extraction."""
    v_idx = config['hyperparameters']['channel_config']['voltage_idx']
    i_idx = config['hyperparameters']['channel_config']['current_idx']
    v_t = block[:, v_idx] * ADC_SCALE * v_step
    i_t = block[:, i_idx] * ADC_SCALE * i_step
    
    res = {}
    # Time Domain (Now fully cycle-averaged)
    res.update(get_time_domain_features(v_t, i_t, config))
    # Frequency Domain (Already cycle-averaged)
    res.update(get_harmonic_analysis(v_t, i_t, config))
    # Wavelet Domain (Transient)
    res.update(get_wavelet_features(i_t, config['features_to_extract']))
    
    return res
