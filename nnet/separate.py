#!/usr/bin/env python

# wujian@2018

import os
import argparse

import torch as th
import numpy as np

from conv_tas_net import ConvTasNet

from libs.utils import load_json, get_logger
from libs.audio import WaveReader, write_wav

logger = get_logger(__name__)


class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir):
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = ConvTasNet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, samps):
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            sps = self.nnet(raw)
            sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
            return sp_samps


def run(args):
    mix_input = WaveReader(args.input, sample_rate=args.fs)
    computer = NnetComputer(args.checkpoint, args.gpu)
    for key, mix_samps in mix_input:
        logger.info("Compute on utterance {}...".format(key))
        spks = computer.compute(mix_samps)
        norm = np.linalg.norm(mix_samps, np.inf)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))
            write_wav(
                os.path.join(args.dump_dir, "spk{}/{}.wav".format(
                    idx + 1, key)),
                samps,
                fs=args.fs)
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--input", type=str, required=True, help="Script for input waveform")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--fs", type=int, default=8000, help="Sample rate for mixture input")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="sps_tas",
        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)