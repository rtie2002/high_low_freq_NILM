# Appliance Availability by House

This document summarizes the appliance channels available in the local REFIT
and UK-DALE datasets. The five target appliances are **kettle, fridge,
dishwasher, washing machine, and microwave**.

`A1` means `Appliance1` in REFIT. `C1` means channel 1 in UK-DALE. A dash (`-`)
means that the dataset does not provide a separate monitored channel. It does
not necessarily mean that the household did not own the appliance.

## Recommended houses for the five-appliance experiment

- **REFIT:** Houses 2, 3, 5, 9, 11, and 20 have all five target appliances.
- **UK-DALE:** Houses 1, 2, and 5 have all five targets in usable channels.
- **UK-DALE House 4 is unsuitable for clean individual ground truth:** kettle
  is combined with a radio, while washing machine and microwave are combined
  with a breadmaker.

## REFIT target-appliance channels

REFIT contains Houses 1-21, excluding House 14.

| House | Kettle | Fridge or fridge-freezer | Dishwasher | Washing machine | Microwave |
|---:|---|---|---|---|---|
| 1 | - | A1 | A6 | A5 | - |
| 2 | A8 | A1 | A3 | A2 | A5 |
| 3 | A9 | A2 | A5 | A6 | A8 |
| 4 | A9 | A1/A3 | - | A4/A5 | A8 |
| 5 | A8 | A1 | A4 | A3 | A7 |
| 6 | A7 | - (freezer only) | A3 | A2 | A6 |
| 7 | A9 | A1 | A6 | A5 | - |
| 8 | A9 | A1 | - | A4 | A8 |
| 9 | A7 | A1 | A4 | A3 | A6 |
| 10 | - | A4 | A6 | A5 | A8 |
| 11 | A7 | A1/A2 | A4 | A3 | A6 |
| 12 | A4 | A1 | - | - | A3 |
| 13 | A9 | - | A4 | A3 | A8 |
| 15 | A8 | A1 | A4 | A3 | A7 |
| 16 | - | A1/A2 | A6 | A5 | - |
| 17 | A8 (shared) | A2 | - | A4 | A7 |
| 18 | - | A1/A3 | A6 | A5 | A9 |
| 19 | A5 | A1 (shared) | - | A2 | A4 |
| 20 | A9 | A1 | A5 | A4 | A8 |
| 21 | A7 (shared) | A1 | A4 | A3 | - |

### REFIT shared-channel warnings

- House 17 A8 includes kettle, toaster, and occasionally other appliances.
- House 19 A1 contains both a fridge and a freezer.
- House 21 A7 contains both kettle and toaster.
- Washer-dryer channels are washing-machine-related but do not provide purely
  washing-machine measurements.
- Computer and television sites can contain several devices.
- Some monitored plugs changed appliances during the REFIT recording period.

## Other REFIT appliances

The following list excludes the five target appliance categories.

| House | Other monitored appliances |
|---:|---|
| 1 | Chest freezer, upright freezer, tumble dryer, computer site, television site, electric heater |
| 2 | Television, toaster, Hi-Fi, oven extractor fan |
| 3 | Toaster, freezer, tumble dryer, television |
| 4 | Freezer, second fridge-freezer, second washing machine, computer site, television site |
| 5 | Tumble dryer/dehumidifier, computer site, television site, toaster |
| 6 | Utility-room freezer, two computer sites, television site, toaster |
| 7 | Two freezers, tumble dryer, television site, toaster |
| 8 | Freezer, dryer, toaster/radio, computer, television site |
| 9 | Washer-dryer, television site, Hi-Fi, electric heater |
| 10 | Blender, freezer, chest freezer, television site, Kenwood mixer |
| 11 | Second fridge-freezer, computer site, router, Hi-Fi |
| 12 | Two television sites, toaster; A7-A9 are unused |
| 13 | Two television sites, unknown appliance, tumble dryer, computer site |
| 15 | Tumble dryer, computer site, television site, toaster |
| 16 | Second fridge-freezer, two electric heaters, computer site, television site, dehumidifier/heater |
| 17 | Freezer, tumble dryer, computer site, television site, bedroom plug site |
| 18 | Garage fridge, freezer, fridge-freezer, washer-dryer, desktop computer, television site |
| 19 | Television site, toaster, bread-maker, lamp, Hi-Fi |
| 20 | Freezer, tumble dryer, computer site, television site |
| 21 | Tumble dryer, food mixer, television, toaster shared with kettle, vivarium, pond pump |

## UK-DALE target-appliance channels

| House | Kettle | Fridge or fridge-freezer | Dishwasher | Washing machine | Microwave |
|---:|---|---|---|---|---|
| 1 | C10 | C12 | C6 | C5 | C13 |
| 2 | C8 | C14 | C13 | C12 | C15 |
| 3 | C2 | - | - | - | - |
| 4 | C3 (shared) | - (freezer only) | - | C6 (shared) | C6 (shared) |
| 5 | C18 | C19 | C22 | C24 (washer-dryer) | C23 |

### UK-DALE shared-channel warnings

- House 4 C3 contains kettle and radio.
- House 4 C6 contains washing machine, microwave, and breadmaker.
- House 5 C24 is a washer-dryer rather than a washing-machine-only channel.

## Other UK-DALE appliances

| House | Other monitored appliances |
|---:|---|
| 1 | Boiler, solar thermal pump, laptop, TV, lights and lamps, HTPC, toaster, office LCD, Hi-Fi, amplifiers, breadmaker, router, soldering iron, network equipment, vacuum, chargers, subwoofer, radios, coffee machine, hair dryer, straighteners, iron, gas oven, data-logger PC, baby monitor, battery charger, office PC, office fan, printer |
| 2 | Laptop, monitor, speakers, servers, router, hard drive, rice cooker, running machine, second laptop, toaster, PlayStation, modem, cooker |
| 3 | Electric heater, laptop, projector |
| 4 | TV/DVD/digibox/lamp, radio shared with kettle, gas boiler, freezer, breadmaker shared with washing machine and microwave |
| 5 | Stereo speakers, desktop computers, hair dryer, televisions/LCDs, treadmill, network storage, server, PS4, steam iron, coffee machine, toaster, home-theatre amplifier, Sky box, oven, electric hob, vacuum cleaner |

## Sources

- REFIT appliance identities: dataset appliance documentation and the local
  preprocessing mapping in `inverse_optimization/config/preprocess/refit.yaml`.
- UK-DALE appliance identities: `labels.dat` inside each local UK-DALE house
  directory.

