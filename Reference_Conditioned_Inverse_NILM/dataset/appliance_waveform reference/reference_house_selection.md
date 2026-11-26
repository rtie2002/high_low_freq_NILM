# Appliance Waveform Reference House Selection

## Purpose

The reference-conditioned inverse non-intrusive load monitoring experiment
needs real appliance waveforms as reference examples. These references are
extracted from houses that **do not contain all five target appliances**.

The five targets are:

1. Kettle
2. Fridge or fridge-freezer
3. Dishwasher
4. Washing machine
5. Microwave

Using incomplete houses for the reference bank keeps them separate from the
houses used for the complete five-appliance experiment.

## Selection rules

A house/channel is selected only when:

- the house does not provide all five target appliances;
- the target appliance has a dedicated monitored channel;
- the channel is not labelled as unknown or unused; and
- the channel does not combine the target with another appliance.

Shared channels are excluded because their waveform is not a clean reference
for one appliance. For example, a `kettle + toaster` channel must not be saved
as a kettle reference.

## REFIT reference houses

`A1` means the `Appliance1` column in the corresponding REFIT house CSV.

| Target appliance | Selected houses and channels |
|---|---|
| Kettle | House 4 A9; House 6 A7; House 7 A9; House 8 A9; House 12 A4; House 13 A9; House 19 A5 |
| Fridge / fridge-freezer | House 1 A1; House 4 A1 and A3; House 7 A1; House 8 A1; House 10 A4; House 12 A1; House 16 A1 and A2; House 17 A2; House 18 A1 and A3; House 21 A1 |
| Dishwasher | House 1 A6; House 6 A3; House 7 A6; House 10 A6; House 13 A4; House 16 A6; House 18 A6; House 21 A4 |
| Washing machine | House 1 A5; House 4 A4 and A5; House 6 A2; House 7 A5; House 8 A4; House 10 A5; House 13 A3; House 16 A5; House 17 A4; House 18 A5; House 19 A2; House 21 A3 |
| Microwave | House 4 A8; House 6 A6; House 8 A8; House 10 A8; House 12 A3; House 13 A8; House 17 A7; House 18 A9; House 19 A4 |

### REFIT channels deliberately excluded

| House/channel | Reason for exclusion |
|---|---|
| House 17 A8 | Contains kettle, toaster, and occasionally other appliances |
| House 19 A1 | Contains both fridge and freezer on one monitor |
| House 21 A7 | Contains both kettle and toaster |
| Washer-dryer channels | Combine washing and drying behaviour; use only if a separate washer-dryer reference category is added later |
| Unknown or unused channels | Appliance identity is not reliable enough for a labelled reference |

### REFIT houses not used as reference providers

Houses 2, 3, 5, 9, 11, 15, and 20 contain all five target appliance
categories. They are therefore reserved for complete five-appliance
experiments and are not used to build this incomplete-house reference bank.

## UK-DALE reference houses

`C2` means channel 2 in the corresponding UK-DALE house directory.

| Target appliance | Selected houses and channels |
|---|---|
| Kettle | House 3 C2 |
| Fridge / fridge-freezer | None from the incomplete houses |
| Dishwasher | None from the incomplete houses |
| Washing machine | None from the incomplete houses |
| Microwave | None from the incomplete houses |

### UK-DALE channels deliberately excluded

| House/channel | Reason for exclusion |
|---|---|
| House 4 C3 | Kettle and radio share the same channel |
| House 4 C6 | Washing machine, microwave, and breadmaker share the same channel |
| House 4 C5 | Freezer, not a fridge or fridge-freezer target channel |

UK-DALE Houses 1, 2, and 5 provide all five target categories and are not used
as incomplete-house reference providers.

## Recommended reference-file organization

See [event_label_definitions.md](event_label_definitions.md) for the timestamp-based
duration rules, the two state labels, and treatment of incomplete event boundaries.

Save all sample rows for one appliance in one long-format CSV. The `dataset`,
`house`, `channel`, and `event_id` columns retain the source information:

```text
appliance_waveform reference/
├── kettle/
│   └── events.csv
├── fridge/
│   └── events.csv
├── dishwasher/
│   └── events.csv
├── washingmachine/
│   └── events.csv
└── microwave/
    └── events.csv
```

Every saved waveform should retain at least the following metadata:

- dataset name;
- house number;
- original appliance channel;
- appliance name;
- start and end timestamps;
- sampling interval;
- ON threshold used for extraction; and
- whether OFF context is included around the ON event.

## Experimental safety rule

A house must never provide reference waveforms for an experiment in which that
same house is evaluated as an unseen validation or testing house. If the
evaluation split changes, rebuild the reference bank after removing all
validation and testing houses. Otherwise, appliance-level information from the
evaluation house would leak into the prediction process.

