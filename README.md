## Pytorch implement of ConvTasNet

### Result

  |  ID   |                   Settings                   | Causal |    Norm     | Param |     Loss      | Si-SDR |
  | :---: | :------------------------------------------: | :---:  | :---------: | :---: | :-----------: | :----: |
  | 11-3c | adam/lr:1e-3/wd:1e-5/100epochs/32-batch/2gpu |   N    |   BN/relu   | 8.75M | -17.59/-15.45 | 14.63  |
  | 11-3a | adam/lr:1e-2/wd:1e-5/100epochs/20-batch/2gpu |   N    |  gLN/relu   |   -   | -16.09/-15.21 | 14.58  |
  | 11-3b | adam/lr:1e-3/wd:1e-5/100epochs/20-batch/2gpu |   N    |  gLN/relu   |   -   | -17.91/-16.54 | 15.87  |
  | 18-3a | adam/lr:1e-2/wd:1e-5/100epochs/32-batch/2gpu |   N    | BN/sigmoid  |   -   | -14.51/-13.40 | 12.62  |
  | 18-3b | adam/lr:1e-2/wd:1e-5/100epochs/32-batch/2gpu |   N    |   BN/relu   |   -   | -17.20/-15.38 | 14.58  |
  | 18-4a | adam/lr:1e-3/wd:1e-5/100epochs/20-batch/2gpu |   N    | gLN/sigmoid |   -   | -17.20/-16.11 | 15.55  |
  | 11-4a | adam/lr:1e-3/wd:1e-5/100epochs/32-batch/2gpu |   Y    |   BN/relu   |   -   | -15.25/-12.47 | 11.42  |
  | 11-4b | adam/lr:1e-3/wd:1e-5/100epochs/24-batch/2gpu |   N    |  cLN/relu   |   -   | -18.72/-16.17 | 15.25  |

### Reference

Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation[J]. arXiv preprint arXiv:1809.07454, 2018.