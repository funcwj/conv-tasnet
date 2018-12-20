# wujian@2018

import os
import sys
import time

from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from utils import get_logger


def offload_egs(egs, device):
    """
    Offload tensor object in egs to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    for key, obj in egs.items():
        # if tensor object, load to gpu
        egs[key] = cuda(obj)
        # if list of tensor
        if isinstance(obj, list):
            egs[key] = list(map(cuda, obj))
    return egs


class AverageMeter(object):
    """
    A simple average meter
    """

    def __init__(self):
        self.val = defaultdict(float)
        self.cnt = defaultdict(int)

    def reset(self):
        self.val.clear()
        self.cnt.clear()

    def add(self, key, value):
        self.val[key] += value
        self.cnt[key] += 1

    def value(self, key):
        if self.cnt[key] == 0:
            return 0
        return self.val[key] / self.cnt[key]

    def sum(self, key):
        return self.val[key]

    def count(self, key):
        return self.cnt[key]


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start


class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.nnet.train()

        stats = AverageMeter()
        timer = SimpleTimer()

        for egs in data_loader:
            # load to gpu
            egs = offload_egs(egs, self.device)

            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)

            stats.add("loss", loss.item())
            loss.backward()

            progress = stats.count("loss")
            if not progress % self.logging_period:
                self.logger.info("Processed {:d} batches...".format(progress))

            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

        return stats.value("loss"), stats.count("loss"), timer.elapsed()

    def eval(self, data_loader):
        self.nnet.eval()

        stats = AverageMeter()
        timer = SimpleTimer()

        with th.no_grad():
            for egs in data_loader:
                egs = offload_egs(egs, self.device)
                loss = self.compute_loss(egs)
                stats.add("loss", loss.item())

        return stats.value("loss"), stats.count("loss"), timer.elapsed()

    def run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # check if save is OK
            self.save_checkpoint(best=False)
            best_loss, _, _ = self.eval(dev_loader)
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
                self.cur_epoch, best_loss))
            while self.cur_epoch < num_epochs:
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        self.optimizer.param_groups[0]["lr"],
                        self.cur_epoch + 1)
                tr_loss, tr_batch, tr_cost = self.train(train_loader)
                stats["tr"] = "train = {:+.4f}({:.2f}s/{:d})".format(
                    tr_loss, tr_cost, tr_batch)
                cv_loss, cv_batch, cv_cost = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.4f}({:.2f}s/{:d})".format(
                    cv_loss, cv_cost, cv_batch)
                stats["scheduler"] = ""
                if cv_loss > best_loss:
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv_loss
                    self.save_checkpoint(best=True)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv_loss)
                # flush scheduler info
                sys.stdout.flush()
                # save checkpoint
                self.cur_epoch += 1
                self.save_checkpoint(best=False)

            self.logger.info(
                "Training for {:d} epoches done!".format(num_epochs))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, egs):
        # spks x n x S
        ests = th.nn.parallel.data_parallel(
            self.nnet, egs["mix"], device_ids=self.gpuid)
        # spks x n x S
        refs = egs["ref"]
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # P x N
        N = egs["mix"].size(0)
        sisnr_mat = th.stack(
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = th.max(sisnr_mat, dim=0)
        # negative si-snr
        return -th.sum(max_perutt) / N