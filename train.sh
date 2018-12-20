#!/usr/bin/env bash

set -eu

cpt_dir=exp/conv_tasnet
epochs=100
batch_size=32

[ $# -ne 2 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1

./nnet/train_tasnet.py \
  --gpu $1 \
  --epochs $epochs \
  --batch-size $batch_size \
  --checkpoint $cpt_dir/$2 \
  > $2.train.log 2>&1
