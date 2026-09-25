# What random mix changes in the code and the training pipeline

Random mix is a change to **how a training sample is built**. The model, the loss, the validation set, and the test set are the same as Clean Version (8w).

One switch turns it on:

```yaml
# config/models/multinilm_fractional_relational.yaml
training:
  random_mix:
    enabled: true
    prob: 0.5
```

`prob: 0.5` means: each time the training loader asks for a window, there is a 50% chance it returns a real window and a 50% chance it returns a newly assembled window. The coin is flipped again every epoch, so the mixed windows are not saved to disk.

---

## 1. Pipeline before this change

```text
training CSV
    -> cut a 1024-sample window from one house
    -> x = that house's aggregate
    -> y = that same window's 5 appliance powers
    -> z = that same window's 5 ON/OFF labels
    -> z-score x and y
    -> model
    -> loss
```

One window index supplies everything. If that house had the kettle and the microwave on together, the model always sees those two labels together.

Validation and test use this same path. They still do.

## 2. Pipeline after this change

```text
training CSV
    -> same window cutting as before
    -> coin flip
         50%: return that real window          (old path)
         50%: build a mixed window             (new path, below)
    -> z-score with the same train-set mean and std
    -> same model
    -> same loss
```

The new path, inside `WindowDataset._random_mix_window`:

```text
draw 6 legal training windows, independently

window 1 -> keep only the kettle column          (power and ON label)
window 2 -> keep only the fridge column
window 3 -> keep only the dishwasher column
window 4 -> keep only the washing-machine column
window 5 -> keep only the microwave column
window 6 -> keep only the background

background = max(aggregate - kettle - fridge - dishwasher
                 - washing machine - microwave, 0)

new aggregate = kettle + fridge + dishwasher
                + washing machine + microwave + background

y = the 5 kept power columns
z = the 5 kept ON labels
x = the new aggregate
```

The label of each appliance is the column that was added into the aggregate. The model still receives one input and five outputs at the same time.

Validation and test call the dataset with mix probability 0, so they stay on the old path.

---

## 3. One timestep, in watts

Old path. One house, one window. The loader returns this row as-is:

| | aggregate | kettle | fridge | dishwasher | washing machine | microwave |
|---|---:|---:|---:|---:|---:|---:|
| power (W) | 2500 | 2000 | 100 | 0 | 0 | 0 |
| ON label | | 1 | 1 | 0 | 0 | 0 |

Background of this row is `2500 - 2100 = 400 W`. That 400 W stays inside the aggregate. It is not a sixth label.

New path. Six different rows are drawn. Only one column is kept from each of the first five:

| source row | column kept | power (W) | ON label |
|---|---|---:|---:|
| row K | kettle | 0 | 0 |
| row F | fridge | 80 | 1 |
| row D | dishwasher | 1200 | 1 |
| row W | washing machine | 0 | 0 |
| row M | microwave | 1500 | 1 |
| row B | background | 300 | none |

The loader returns:

| | aggregate | kettle | fridge | dishwasher | washing machine | microwave |
|---|---:|---:|---:|---:|---:|---:|
| power (W) | 3080 | 0 | 80 | 1200 | 0 | 1500 |
| ON label | | 0 | 1 | 1 | 0 | 1 |

`3080 = 0 + 80 + 1200 + 0 + 1500 + 300`.

This combination did not occur in any one house. The kettle label no longer arrives with whatever else was on in the kettle's house. The five targets are still returned together, in the same tensor the model already expects.

After this table, both paths do the same z-score and the same crop. Current yaml has input length = output length = 1024, so the crop keeps the whole window.

---

## 4. Where each step lives

The run order is:

| Step | File | What it does |
|---|---|---|
| 1 | `config/models/multinilm_fractional_relational.yaml` | Sets `enabled: true` and `prob: 0.5`. |
| 2 | `data/dataloader.py`, `get_random_mix_prob` | Reads that block. `enabled: false` returns probability 0, which is the old pipeline. A probability outside `(0, 1]` raises an error. |
| 3 | `data/dataloader.py`, `NILMDataLoader.__init__` | Stores the probability on the loader. |
| 4 | `data/dataloader.py`, `NILMDataLoader._make_window_dataset` | Passes the probability into the **train** dataset. Passes `0.0` for validation and every test scenario. |
| 5 | `data/dataloader.py`, `WindowDataset.__init__` | If the probability is above 0, stores two extra arrays in watts, before z-score: `targets_watts` (the five submeters) and `residual_watts` (the background, clipped at 0 W). |
| 6 | `data/dataloader.py`, `WindowDataset.__getitem__` | Draws a uniform random number. Below 0.5, copies one real window. Otherwise calls `_random_mix_window`. |
| 7 | `data/dataloader.py`, `WindowDataset._random_mix_window` | Draws the six starts, stacks the five columns, adds the background, z-scores the new aggregate and the five powers. |
| 8 | `runner.py` | Prints `Random mix p=0.5 (train only)` at startup. It does not mix anything itself. |
| 9 | `model/MultiNILM.py`, `model/MultiNILM_loss.py` | Unchanged. They still receive `(x, y, z)` and still emit five power channels and five state channels. |

The six starts are taken from `self.indices`. Those indices already stop at a house boundary or a time gap, so one copied column is 1024 consecutive samples from a single house. The six columns may come from six different houses.

Shapes returned to the training loop, per sample:

| Tensor | Shape | Contents on a mixed draw |
|---|---|---|
| `x` | `(1024, 1)` | z-scored mixed aggregate |
| `y` | `(1024, 5)` | z-scored copied submeters, order kettle, fridge, dishwasher, washing machine, microwave |
| `z` | `(1024, 5)` | copied CSV `*_on` labels, same order |

The batch size is still 64. A batch can contain a mixture of real windows and mixed windows.

---

## 5. What you should see when you train

At startup the summary line is:

```text
Random mix    p=0.5 (train only)
```

If that line says `off`, the yaml block was not loaded and every training window is real.

No new checkpoint key, no new loss term, and no new metric. The test metrics are still computed on real UK-DALE House 2 and REFIT House 20 windows. Compare those metrics with Clean Version (8w). That run used this same model with the mix switch off.

---

## 6. Where the idea comes from

The code follows the additive NILM measurement, aggregate = sum of target appliances + other load (Hart, *Proceedings of the IEEE*, 1992). The training trick is to rebuild that sum from pieces of real recordings:

- Kelly and Knottenbelt, "Neural NILM: Deep Neural Networks Applied to Energy Disaggregation", ACM BuildSys 2015 (arXiv:1507.06594). This is the closest precedent. They trained on a **50:50 mix of real and synthetic aggregates**, built the synthetic ones by adding real appliance activations, reported that the synthetic half acts as a regulariser and improves generalisation to unseen houses, and validated and tested on real data only. `prob: 0.5`, the train-only switch, and the unseen-house goal follow that recipe. Two differences: they placed whole activations into an empty window, one network per appliance, while this code copies whole 1024-sample windows per appliance and keeps a real background residual under them.
- Rafiq et al., *IEEE Transactions on Smart Grid*, 2021: make a synthetic aggregate whose submeter labels are exactly the pieces you added, so the model has to work on houses it was not trained on.
- Kamyshev et al., arXiv:2106.02352 (SNS / COLD): choose which loads are on together by a random draw, then add their measured signatures.
- Nour et al., HAL hal-03513298, 2021: build the multi-load signal by adding real recordings.

This repository applies that sum inside the existing multi-appliance loader. It does not add a new network, and it does not blend labels the way mixup does (Zhang et al., ICLR 2018). A copied ON label stays 0 or 1.
