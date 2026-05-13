import pandas as pd
import numpy as np

df = pd.read_csv('high_frequency_data_extract/output/washingmachine_house2_1374447600.csv')

print("=== SHAPE ===")
print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

# ── Ratio check ────────────────────────────────────────────────────────────────
print("\n=== RATIO (aggregate / P_active) ===")
df['ratio'] = df['aggregate'] / df['P_active']
print(f"Mean:  {df['ratio'].mean():.4f}")
print(f"Std:   {df['ratio'].std():.4f}")
print(f"Min:   {df['ratio'].min():.4f}")
print(f"Max:   {df['ratio'].max():.4f}")

# ── ON/OFF distribution ────────────────────────────────────────────────────────
print("\n=== WASHING MACHINE ON/OFF LABEL DISTRIBUTION ===")
print(df['on_off'].value_counts())
wm_on_pct = (df['on_off'] == 1).mean() * 100
print(f"→ {wm_on_pct:.1f}% of this hour the washing machine is ON")

# ── ON/OFF Transitions ─────────────────────────────────────────────────────────
print("\n=== ON/OFF TRANSITIONS ===")
df['on_off_change'] = df['on_off'].diff().abs()
transitions = df[df['on_off_change'] > 0][['readable_time', 'P_active', 'I_rms', 'aggregate', 'washingmachine_power', 'on_off']]
print(transitions.to_string())
print(f"\nTotal transitions: {len(transitions)}")

# ── WM Power when ON vs OFF ────────────────────────────────────────────────────
print("\n=== WASHING MACHINE POWER STATS ===")
on_mask  = df['on_off'] == 1
off_mask = df['on_off'] == 0
print(f"When ON  → wm_power: mean={df.loc[on_mask,'washingmachine_power'].mean():.1f}W  "
      f"max={df.loc[on_mask,'washingmachine_power'].max():.1f}W")
print(f"When OFF → wm_power: mean={df.loc[off_mask,'washingmachine_power'].mean():.1f}W  "
      f"max={df.loc[off_mask,'washingmachine_power'].max():.1f}W")
print(f"When ON  → P_active (HF): mean={df.loc[on_mask,'P_active'].mean():.1f}W  "
      f"max={df.loc[on_mask,'P_active'].max():.1f}W")
print(f"When OFF → P_active (HF): mean={df.loc[off_mask,'P_active'].mean():.1f}W  "
      f"max={df.loc[off_mask,'P_active'].max():.1f}W")

# ── Delta-P direction correlation ──────────────────────────────────────────────
print("\n=== DELTA-P DIRECTION CORRELATION ===")
df['dP_hf'] = df['P_active'].diff()
df['dP_lf'] = df['aggregate'].diff()
same_dir = ((df['dP_hf'] * df['dP_lf']) > 0).sum()
total = df['dP_hf'].notna().sum()
print(f"Same direction: {same_dir}/{total} = {100*same_dir/total:.1f}%")

# ── HF feature change at WM events ────────────────────────────────────────────
print("\n=== KEY HF FEATURES: ON vs OFF comparison ===")
features = ['I_rms', 'THDI', 'I_skew', 'I_kurt', 'PF', 'DWT_E0']
for f in features:
    if f in df.columns:
        on_val  = df.loc[on_mask, f].mean()
        off_val = df.loc[off_mask, f].mean()
        print(f"  {f:<12}  ON={on_val:.4f}   OFF={off_val:.4f}   ratio={on_val/off_val:.2f}x")

# ── Show rows around first ON transition ──────────────────────────────────────
if len(transitions) > 0:
    first_t = transitions.iloc[0]['readable_time']
    idx = df[df['readable_time'] == first_t].index[0]
    window = df.iloc[max(0, idx-2):idx+4][['readable_time', 'P_active', 'I_rms', 'THDI', 'PF',
                                           'aggregate', 'washingmachine_power', 'on_off']]
    print(f"\n=== TRANSITION WINDOW (around {first_t}) ===")
    print(window.to_string())
