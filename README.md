## ConvTasNet

A PyTorch implementation of the [TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation](https://arxiv.org/abs/1809.07454)

### Requirements

* scipy 1.1.0
* torch 1.0.0
* numpy 1.14.3

### Usage

* training: configure [conf.py](nnet/conf.py) and start [train.sh](train.sh)

* inference
```
./nnet/time_domain_separate.py /path/to/checkpoint --input /path/to/mix.scp --gpu 0 > separate.log 2>&1 &
```

* evaluate
```
./nnet/compute_si_snr.py /path/to/ref_spk1.scp,/path/to/ref_spk2.scp /path/to/inf_spk1.scp,/path/to/inf_spk2.scp
```

### Result (on best configuratures in the paper)

  |  ID   |                   Settings                   | Causal |    Norm     | Param |     Loss      | Si-SDR |
  | :---: | :------------------------------------------: | :---:  | :---------: | :---: | :-----------: | :----: |
  | 01-3c | adam/lr:1e-3/wd:1e-5/100epochs/32-batch/2gpu |   N    |   BN/relu   | 8.75M | -17.59/-15.45 | 14.63  |
  | 01-3a | adam/lr:1e-2/wd:1e-5/100epochs/20-batch/2gpu |   N    |  gLN/relu   |   -   | -16.09/-15.21 | 14.58  |
  | 01-3b | adam/lr:1e-3/wd:1e-5/100epochs/20-batch/2gpu |   N    |  gLN/relu   |   -   | -17.91/-16.54 | 15.87  |
  | 02-3a | adam/lr:1e-2/wd:1e-5/100epochs/32-batch/2gpu |   N    | BN/sigmoid  |   -   | -14.51/-13.40 | 12.62  |
  | 02-3b | adam/lr:1e-2/wd:1e-5/100epochs/32-batch/2gpu |   N    |   BN/relu   |   -   | -17.20/-15.38 | 14.58  |
  | 02-4a | adam/lr:1e-3/wd:1e-5/100epochs/20-batch/2gpu |   N    | gLN/sigmoid |   -   | -17.20/-16.11 | 15.55  |
  | 01-4a | adam/lr:1e-3/wd:1e-5/100epochs/32-batch/2gpu |   Y    |   BN/relu   |   -   | -15.25/-12.47 | 11.42  |
  | 01-4b | adam/lr:1e-3/wd:1e-5/100epochs/24-batch/2gpu |   N    |  cLN/relu   |   -   | -18.72/-16.17 | 15.25  |

### Reference

Luo Y, Mesgarani N. TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation[J]. arXiv preprint arXiv:1809.07454, 2018.