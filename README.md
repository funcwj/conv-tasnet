## Pytorch implement of ConvTasNet

### Result

1. non-causal

  |  ID   |                         Settings                         | Param |     Loss      | Si-SDR |
  | :---: | :------------------------------------------------------: | :---: | :-----------: | :----: |
  | 11-3c |     adam/lr:1e-3/wd:1e-5/BN/100epochs/32-batch/2gpu      | 8.75M | -17.59/-15.45 | 14.63  |
  | 11-3a |     adam/lr:1e-2/wd:1e-5/gLN/100epochs/32-batch/2gpu     |   -   | -16.09/-15.21 | 14.58  |
  | 11-3b |     adam/lr:1e-3/wd:1e-5/gLN/100epochs/32-batch/2gpu     |   -   | -17.91/-16.54 | 15.87  |
  | 18-3a | adam/lr:1e-2/wd:1e-5/BN/100epochs/32-batch/2gpu/sigmoid  |   -   | -14.51/-13.40 | 12.62  |
  | 18-3b |     adam/lr:1e-2/wd:1e-5/BN/100epochs/32-batch/2gpu      |   -   | -17.20/-15.38 | 14.58  |

### Reference

Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation[J]. arXiv preprint arXiv:1809.07454, 2018.