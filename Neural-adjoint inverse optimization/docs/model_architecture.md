# CNN architecture report

Generated from `stage1.yaml`.

| Item | Value |
| --- | --- |
| Input power | `[B, 5, 1024]` |
| Encoder single-trace input | `[B, 1, 1024]` |
| Latent code | `[B, 5, 2, 128]` |
| Decoder output | `[B, 5, 1024]` |
| Downsampling factor | 8 |
| Trainable parameters | 9,731 |

## Encoder

`Sequential(
  (0): Conv1d(1, 32, kernel_size=(4,), stride=(2,), padding=(1,))
  (1): GELU(approximate='none')
  (2): Conv1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
  (3): GELU(approximate='none')
  (4): Conv1d(32, 2, kernel_size=(4,), stride=(2,), padding=(1,))
)`

## Decoder

`Sequential(
  (0): ConvTranspose1d(7, 32, kernel_size=(4,), stride=(2,), padding=(1,))
  (1): GELU(approximate='none')
  (2): ConvTranspose1d(32, 32, kernel_size=(4,), stride=(2,), padding=(1,))
  (3): GELU(approximate='none')
  (4): ConvTranspose1d(32, 1, kernel_size=(4,), stride=(2,), padding=(1,))
)`

The shared conditional decoder concatenates one fixed five-dimensional appliance identity vector to each latent code before decoding. The final Softplus enforces nonnegative power.
