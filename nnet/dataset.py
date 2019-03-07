# wujian@2018

import random
import torch as th
import numpy as np

from audio import WaveReader
from torch.utils.data.dataloader import default_collate


def make_dataloader(train=True,
                    data_kwargs=None,
                    chunk_size=32000,
                    batch_size=16,
                    cache_size=32):
    perutt_loader = PeruttLoader(shuffle=train, **data_kwargs)
    return DataLoader(
        perutt_loader,
        train=train,
        chunk_size=chunk_size,
        batch_size=batch_size,
        cache_size=cache_size)


class PeruttLoader(object):
    """
    Per Utterance Loader
    """

    def __init__(self,
                 shuffle=True,
                 mix_scp="",
                 ref_scp=None,
                 sample_rate=8000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = [WaveReader(y, sample_rate=sample_rate) for y in ref_scp]
        self.shuffle = shuffle

    def _make_ref(self, key):
        for reader in self.ref:
            if key not in reader:
                return None
        return [reader[key] for reader in self.ref]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.mix.index_keys)
        for key, mix in self.mix:
            ref = self._make_ref(key)
            if ref is not None:
                eg = dict()
                eg["mix"] = mix.astype(np.float32)
                eg["ref"] = [r.astype(np.float32) for r in ref]
                yield eg


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """

    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[s:s + self.chunk_size] for ref in eg["ref"]]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = [
                np.pad(ref, (0, P), "constant") for ref in eg["ref"]
            ]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level PIT
    """

    def __init__(self,
                 perutt_loader,
                 chunk_size=32000,
                 batch_size=16,
                 cache_size=16,
                 train=True):
        self.loader = perutt_loader
        self.cache_size = cache_size * batch_size
        self.batch_size = batch_size
        self.splitter = ChunkSplitter(
            chunk_size, train=train, least=chunk_size // 2)

    def _fetch_batch(self):
        while True:
            if len(self.load_list) >= self.cache_size:
                break
            try:
                eg = next(self.load_iter)
                cs = self.splitter.split(eg)
                self.load_list.extend(cs)
            except StopIteration:
                self.stop_iter = True
                break
        random.shuffle(self.load_list)
        N = len(self.load_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(self.load_list[s:s + self.batch_size])
            blist.append(batch)
        # update load_list
        rn = N % self.batch_size
        if rn:
            # add last batch
            if self.stop_iter and rn >= 4:
                last = default_collate(self.load_list[-rn:])
                blist.append(last)
            else:
                self.load_list = self.load_list[-rn:]
        else:
            self.load_list = []
        return blist

    def __iter__(self):
        # reset flags
        self.load_iter = iter(self.loader)
        self.stop_iter = False
        self.load_list = []

        while not self.stop_iter:
            bs = self._fetch_batch()
            for obj in bs:
                yield obj