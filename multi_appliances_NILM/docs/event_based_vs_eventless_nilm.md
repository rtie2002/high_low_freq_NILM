# Event-Based vs. Event-Less NILM

| Aspect | Event-based NILM | Event-less NILM (this project) |
|---|---|---|
| Core idea | • Detect switching events, then classify them<br>• Unit: one event (instant) | • Map aggregate directly to appliance power/state<br>• Unit: every time step / window |
| Input → output | • Input: only short segments around detected events<br>• Output: one class per event (appliance + ON/OFF)<br>• Energy ≈ ΔP × ON duration (paired events) | • Input: all aggregate data (sliding windows)<br>• Output: per-sample power (regression) + ON/OFF (classification)<br>• Energy = sum of predicted power |
| Features & labels | • Features: ΔP/ΔQ, transients, harmonics, V-I trajectory<br>• Labels: event list (time, appliance, ON/OFF, ΔP)<br>• Needs manual labelling | • Features: learned from power series<br>• Labels: submetered power + thresholded ON/OFF<br>• Labels obtained automatically |
| Data & metrics | • Mostly high frequency (kHz V/I)<br>• Event detection P/R/F1, classification accuracy | • Mostly low frequency (1 s – 1 min)<br>• MAE, SAE, energy error, per-sample F1 |
| Pros | • Low computation<br>• Strong high-rate features<br>• Interpretable per event | • No event detector<br>• Handles variable loads and overlaps<br>• Works on smart-meter data |
| Cons | • Detector errors cascade<br>• Weak on variable loads and overlapping events | • Higher computation<br>• Similar appliances hard to separate at low rate |
| Datasets | • Scarce (5 of 26; only 2 with event labels)<br>• BLUED, SustDataED, BLOND-50, NILMPEds<br>• Transient libraries: PLAID, WHITED, COOLL | • Abundant (21 of 26)<br>• UK-DALE, REFIT, REDD, AMPds, ECO, Dataport |

**Key sources:** Pereira & Nunes, *WIREs DMKD* 2018 ([doi](https://doi.org/10.1002/widm.1265)); BLUED ([link](https://huggingface.co/datasets/INFERLab/BLUED)); NILMPEds, *Data* 2019 ([doi](https://doi.org/10.3390/data4030127)); UK-DALE, *Sci. Data* 2015 ([link](https://www.nature.com/articles/sdata20157)); REFIT, *Sci. Data* 2017 ([link](https://www.nature.com/articles/sdata2016122)); PLAID, *Sci. Data* 2020 ([doi](https://doi.org/10.1038/s41597-020-0389-7)); Lu et al., *Energies* 2023 ([doi](https://doi.org/10.3390/en16020939)).
