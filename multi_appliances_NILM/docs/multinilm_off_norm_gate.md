# MultiNILM OFF-norm gate (slide notes)

**Issue:** OFF predictions sat at **mean (W)** (fridge 50 W), not 0 W.

**Cause:** z-score uses \(\tilde{P}=(P-\mu)/\sigma\). OFF target is \(\tilde{P}=-\mu/\sigma\), but old gate gave \(\tilde{P}=0\) → \(P=\mu\).

**Old:** \(\hat{P} = g \cdot \hat{P}_{\mathrm{raw}}\) — when \(g\to0\), \(\hat{P}\to0\) → **spike at \(\mu\) W**

**Fix:** \(\hat{P} = g\,\hat{P}_{\mathrm{raw}} + (1-g)\,\mathrm{off\_norm}\), where \(\mathrm{off\_norm}=-\mu/\sigma\)

| \(g\) | Result |
|-------|--------|
| 0 (OFF) | \(\hat{P}=\mathrm{off\_norm}\) → **0 W** |
| 1 (ON)  | \(\hat{P}=\hat{P}_{\mathrm{raw}}\) → ON power |

**Code:** `MultiNILM.py` / `TransferNILM.py` (blend) + `config.py` (off_norm from yaml stats). **Retrain** required.

**Baseline note:** Author `transfer_learning_multi-appliance` uses `power = raw * sigmoid(state)` only. OFF-norm blend is a deliberate z-score fix; retrain Transfer after sync — author checkpoints are not bit-identical forward.
