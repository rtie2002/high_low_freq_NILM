# Multi-appliance NILM

Live model: **MultiNILM fractional + relation attention**.

```powershell
python main.py --mode train_evaluate --model multinilm_fractional --experiment config/experiment_ukdale.yaml --model-config config/models/multinilm_fractional_relational.yaml
```

Click Run on `main.py` for the same defaults.

## Read these files

| File | What it is |
|------|------------|
| `config/models/multinilm_fractional_relational.yaml` | knobs |
| `model/MultiNILM.py` | architecture + GL front-end (`forward` first) |
| `model/MultiNILM_loss.py` | power / state / optional DA |
| `adapters/multinilm.py` | yaml → model → one batch |
| `adapters/dataloader.py` | CSV → windows → z-score |
| `runner.py` | train / eval loop |

```text
CSV watts → dataloader → GL channels → stem+TCN → A heads
         → relation attention → power, state logits → loss
```

`MATNILM` / `MATUDA` / `TransferNILM` are other papers, not this model.
`scripts/` is dataset prep. `docs/` is equation notes.
