## ConvTasNet

A PyTorch implementation of the [TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation](https://arxiv.org/abs/1809.07454)

### Requirements

see [requirements.txt](requirements.txt)

### Usage

* training: configure [conf.py](nnet/conf.py) and run [train.sh](train.sh)

* inference
```bash
./nnet/separate.py /path/to/checkpoint --input /path/to/mix.scp --gpu 0 > separate.log 2>&1 &
```

* evaluate
```bash
./nnet/compute_si_snr.py /path/to/ref_spk1.scp,/path/to/ref_spk2.scp /path/to/inf_spk1.scp,/path/to/inf_spk2.scp
```

### Result (on best configuratures in the paper)

  |  ID   |             Settings               | Causal |    Norm     | Param |     Loss      | Si-SDR |
  | :---: | :--------------------------------: | :---:  | :---------: | :---: | :-----------: | :----: |
  |   0   | adam/lr:1e-3/wd:1e-5/32-batch/2gpu |   N    |   BN/relu   | 8.75M | -17.59/-15.45 | 14.63  |
  |   1   | adam/lr:1e-2/wd:1e-5/20-batch/2gpu |   N    |  gLN/relu   |   -   | -16.09/-15.21 | 14.58  |
  |   2   | adam/lr:1e-3/wd:1e-5/20-batch/2gpu |   N    |  gLN/relu   |   -   | -17.91/-16.54 | 15.87  |
  |   3   | adam/lr:1e-2/wd:1e-5/32-batch/2gpu |   N    | BN/sigmoid  |   -   | -14.51/-13.40 | 12.62  |
  |   4   | adam/lr:1e-2/wd:1e-5/32-batch/2gpu |   N    |   BN/relu   |   -   | -17.20/-15.38 | 14.58  |
  |   5   | adam/lr:1e-3/wd:1e-5/20-batch/2gpu |   N    | gLN/sigmoid |   -   | -17.20/-16.11 | 15.55  |
  |   6   | adam/lr:1e-3/wd:1e-5/32-batch/2gpu |   Y    |   BN/relu   |   -   | -15.25/-12.47 | 11.42  |
  |   7   | adam/lr:1e-3/wd:1e-5/24-batch/2gpu |   N    |  cLN/relu   |   -   | -18.72/-16.17 | 15.25  |

### Reference

Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation[J]. arXiv preprint arXiv:1809.07454, 2018.