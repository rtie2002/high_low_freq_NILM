# Appliance and Background Models for Non-Intrusive Load Monitoring

The five target appliances can share a convolutional neural network (CNN) or use one CNN per appliance. Background power covers unmonitored loads and discrepancies between the mains and the five submeters.

| Design | Stage 1 — Training | Stage 2 — Aggregate-only inference | Role in Paper 1 |
| --- | --- | --- | --- |
| **Six-channel aggregate predictor** | Feed five appliance traces and the training residual into a network. Predict the mains. | Freeze the network and optimise six candidate traces to match the mains. | The network can minimise training error by learning addition. Aggregate fit alone cannot identify individual sources. |
| **Five-channel CNN + simple background constraint** | Train the CNN to reconstruct the five appliance traces. | Optimise five appliance codes and a low-dimensional background, such as a constant per window. | First baseline. A simple background may miss sudden changes from unmonitored loads. |
| **Five-channel CNN + learned background model** | Train the appliance CNN and a separate model on training residuals, computed as mains minus the five submeter traces. | Freeze both models and jointly estimate appliance and background codes from the mains. | Follow-up experiment. A flexible background can capture more patterns and may also absorb target-appliance events. |
| **Five independent appliance CNNs** | Train one autoencoder per appliance using its own submeter traces. | Freeze all five decoders and jointly optimise five appliance codes and the background from one mains signal. | Compare against the shared CNN with the same background treatment and optimisation budget. Report total parameter counts. |

**Evaluation:** Compare target-appliance ON-state mean absolute error, F1 score, false positives, and missed events on the same validation windows.
