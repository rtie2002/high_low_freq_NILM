import pandas as pd

df = pd.read_csv('high_frequency_data_extract/output/fridge_house2_1374447600.csv')

print("=== SHAPE ===")
print(f"Rows: {len(df)}, Cols: {len(df.columns)}")

# ── Ratio check ────────────────────────────────────────────────────────────────
print("\n=== RATIO (aggregate / P_active) ===")
df['ratio'] = df['aggregate'] / df['P_active']
print(f"Mean:  {df['ratio'].mean():.4f}")
print(f"Std:   {df['ratio'].std():.4f}")
print(f"Min:   {df['ratio'].min():.4f}")
print(f"Max:   {df['ratio'].max():.4f}")
print("→ If Std is small (<0.2), the offset is systematic (calibration), NOT time mismatch.")

# ── Spike time alignment ───────────────────────────────────────────────────────
print("\n=== TIME ALIGNMENT: SPIKE at 00:23:00 (big appliance event) ===")
for t in ['2013-07-22 00:22:54', '2013-07-22 00:23:00', '2013-07-22 00:23:06']:
    row = df[df['readable_time'] == t]
    if not row.empty:
        r = row.iloc[0]
        print(f"  [{t}]  P_active={r['P_active']:.1f}W  I_rms={r['I_rms']:.4f}A  "
              f"aggregate={r['aggregate']:.1f}W  fridge_power={r['fridge_power']:.1f}W  on_off={int(r['on_off'])}")

print("\n→ KEY TEST: Does P_active spike at EXACTLY the same row as aggregate?")
print("  If yes → time is ALIGNED. The ~2x offset is a calibration constant.")
print("  If no  → time is MISMATCHED.")

# ── Fridge ON/OFF transitions ──────────────────────────────────────────────────
print("\n=== FRIDGE ON/OFF TRANSITIONS ===")
df['on_off_change'] = df['on_off'].diff().abs()
transitions = df[df['on_off_change'] > 0][['readable_time', 'P_active', 'I_rms', 'aggregate', 'fridge_power', 'on_off']]
print(transitions.to_string())
print(f"\nTotal transitions: {len(transitions)}")

# ── Delta-P correlation ────────────────────────────────────────────────────────
print("\n=== DELTA-P CORRELATION (HF vs LF change direction) ===")
df['dP_hf'] = df['P_active'].diff()
df['dP_lf'] = df['aggregate'].diff()
same_direction = ((df['dP_hf'] * df['dP_lf']) > 0).sum()
total = df['dP_hf'].notna().sum()
print(f"HF and LF change in SAME direction: {same_direction}/{total} = {100*same_direction/total:.1f}%")
print("→ If >70%, time alignment is CONFIRMED.")
