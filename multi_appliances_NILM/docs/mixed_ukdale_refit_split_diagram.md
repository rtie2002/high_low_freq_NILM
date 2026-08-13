# Mixed UK-DALE + REFIT 3-Week Split

This is the idea behind `experiment_mixed_ukdale_refit_3w.yaml` and
`scripts/prepare_mixed_ukdale_refit_3week_split.py`.

## Big Idea

You are not simply joining the full UK-DALE and REFIT datasets.

Instead, you:

1. Pick active 3-week blocks per house.
2. Use some houses as labeled source houses.
3. Split each source house block into 80% train and 20% validation.
4. Hold out one UK-DALE house and one REFIT house as test houses.
5. Train one model on the mixed labeled source data.
6. When domain adaptation is enabled, use the test split as unlabeled target data for feature alignment.

## House-Level Split

```text
                         ALL AVAILABLE HOUSE CSVs
                                  |
              +-------------------+-------------------+
              |                                       |
          UK-DALE                                  REFIT
              |                                       |
      +-------+-------+                  +------------+-------------+
      |               |                  |                          |
  source houses    test house        source houses               test house
  H1, H5           H2                H2, H3, H5, H9, H11          H20
      |               |                  |                          |
      +-------+-------+------------------+-------------+------------+
              |                                      |
       labeled source pool                    held-out test pool
```

The selected source houses are:

```text
UK-DALE source: H1, H5
REFIT source:   H2, H3, H5, H9, H11
```

The held-out test houses are:

```text
UK-DALE test: H2
REFIT test:   H20
```

## Per-House 3-Week Block Selection

For every house, the script slides a 3-week window over the timeline.

```text
full house timeline
|--------------------------------------------------------------------|

candidate 3-week windows, step = 1 day
|---------------------|
  |---------------------|
    |---------------------|
      |---------------------|
        ...
```

Each candidate block is checked for:

- enough sample coverage
- enough ON events for each appliance
- enough ON minutes for non-fridge appliances
- enough fridge activity by events or ON fraction

Then the script chooses the strongest valid block using a maximin score:

```math
score = \min_a \frac{n_\text{events}(a)}{\text{min_events}(a)}
```

Meaning:

> choose the block where the weakest appliance is still as active as possible.

This avoids picking a 3-week window where, for example, kettle and fridge are active but dishwasher or washing machine is almost missing.

## Source House Flow

For each source house, the chosen 3-week block is split by time:

```text
selected 3-week active block
|--------------------------------------------------|
|                    80%                           |       20%
|-------------------- train -----------------------|--- validation ---|
```

So source houses produce labeled train and labeled validation rows:

```text
UK-DALE H1 selected block  -> first 80% train, last 20% val
UK-DALE H5 selected block  -> first 80% train, last 20% val

REFIT H2 selected block    -> first 80% train, last 20% val
REFIT H3 selected block    -> first 80% train, last 20% val
REFIT H5 selected block    -> first 80% train, last 20% val
REFIT H9 selected block    -> first 80% train, last 20% val
REFIT H11 selected block   -> first 80% train, last 20% val
```

Then all source train parts are concatenated:

```text
UK-DALE H1 train
UK-DALE H5 train
REFIT H2 train
REFIT H3 train
REFIT H5 train
REFIT H9 train
REFIT H11 train
        |
        v
datasets/mixed_ukdale_refit_3w/training/multi_appliance_training.csv
```

And all source validation parts are concatenated:

```text
UK-DALE H1 val
UK-DALE H5 val
REFIT H2 val
REFIT H3 val
REFIT H5 val
REFIT H9 val
REFIT H11 val
        |
        v
datasets/mixed_ukdale_refit_3w/validating/multi_appliance_validating.csv
```

## Test House Flow

For test houses, the selected active 3-week block is not split into train/val.
The whole block goes into the test CSV.

```text
UK-DALE H2 selected 3-week block
REFIT H20 selected 3-week block
        |
        v
datasets/mixed_ukdale_refit_3w/testing/multi_appliance_testing.csv
```

## Full Dataset Construction Diagram

```text
UK-DALE H1  -- active 3w block -- 80% --> train --+
                                  20% --> val ----+ 
UK-DALE H5  -- active 3w block -- 80% --> train --+ 
                                  20% --> val ----+
                                                   |
REFIT H2    -- active 3w block -- 80% --> train --+--> mixed TRAIN CSV
                                  20% --> val ----+--> mixed VAL CSV
REFIT H3    -- active 3w block -- 80% --> train --+
                                  20% --> val ----+
REFIT H5    -- active 3w block -- 80% --> train --+
                                  20% --> val ----+
REFIT H9    -- active 3w block -- 80% --> train --+
                                  20% --> val ----+
REFIT H11   -- active 3w block -- 80% --> train --+
                                  20% --> val ----+

UK-DALE H2  -- active 3w block ----------------------> mixed TEST CSV
REFIT H20   -- active 3w block ----------------------> mixed TEST CSV
```

## Training Flow

The model sees five common appliances:

```text
kettle, fridge, dishwasher, washingmachine, microwave
```

The mixed experiment uses one shared normalization fitted from the mixed training CSV:

```text
mixed TRAIN only
      |
      v
shared z-score statistics
      |
      +--> apply to train
      +--> apply to validation
      +--> apply to test
```

Training without domain adaptation is:

```text
mixed labeled TRAIN
        |
        v
MultiNILM / MultiNILM-Fractional
        |
        v
power loss + ON/OFF state loss
        |
        v
validate on mixed VAL
        |
        v
final evaluation on mixed TEST
```

With domain adaptation enabled in `multinilm_fractional.yaml`, the flow becomes:

```text
                   labeled source batch
              from mixed TRAIN CSV
                       |
                       v
aggregate -> fractional front-end -> MultiNILM features -> power/state heads
                       |                                      |
                       |                                      v
                       |                                task loss
                       |
                       v
                 source features


                   unlabeled target batch
              from mixed TEST CSV
                       |
                       v
aggregate -> fractional front-end -> MultiNILM features
                       |
                       v
                 target features


source features + target features
          |
          v
domain alignment loss: MMD / CORAL / both
          |
          v
total loss = task loss + domain alignment pressure
```

In your config:

```yaml
domain_adaptation:
  enabled: true
  target_split: test

loss:
  domain_method: both
  domain_mu: 0.4
  domain_mix: convex
  domain_scale: equal
  lambda_domain: 0.3
```

So the conceptual training objective is:

```math
L =
(1-\lambda) L_\text{task}
+ \lambda L_\text{domain}
```

where:

```math
\lambda = 0.3
```

and:

```math
L_\text{domain}
=
\mu L_\text{MMD}
+ (1-\mu)L_\text{CORAL}
```

with:

```math
\mu = 0.4
```

## The Main Research Idea

The dataset mix is doing two things at once:

1. More labeled source diversity:

```text
UK-DALE source houses + REFIT source houses
= broader appliance behavior and household variation
```

2. Cross-domain robustness:

```text
train on mixed source houses
align features toward held-out target houses
test on unseen UK-DALE H2 and REFIT H20
```

So the core idea is:

> learn appliance signatures from multiple homes and both datasets, while forcing the model features to be less tied to one dataset or one house.

## One-Line Summary

You build a mixed source domain from active 3-week blocks of UK-DALE H1/H5 and REFIT H2/H3/H5/H9/H11, validate on the last 20% of those same source blocks, and test on held-out active 3-week blocks from UK-DALE H2 and REFIT H20; with DA enabled, the test split is also used unlabeled for feature alignment during training.
