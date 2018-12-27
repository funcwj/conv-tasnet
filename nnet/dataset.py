# wujian@2018

import random
import torch as th
import numpy as np

from audio import WaveReader


def make_dataloader(shuffle=True,
                    data_kwargs=None,
                    chunk_size=32000,
                    batch_size=16,
                    cache_size=256):
    perutt_loader = PeruttLoader(shuffle=shuffle, **data_kwargs)
    return DataLoader(
        perutt_loader,
        chunk_size=chunk_size,
        batch_size=batch_size,
        cache_size=cache_size)


class PeruttLoader(object):
    """
    Per Utterance Loader
    """

    def __init__(self,
                 shuffle=True,
                 audio_x="",
                 audio_y=None,
                 sample_rate=8000):
        self.audio_x = WaveReader(audio_x, sample_rate=sample_rate)
        self.audio_y = [
            WaveReader(y, sample_rate=sample_rate) for y in audio_y
        ]
        self.shuffle = shuffle

    def _make_ref(self, key):
        for reader in self.audio_y:
            if key not in reader:
                return None
        return [reader[key] for reader in self.audio_y]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.audio_x.index_keys)
        for key, mix in self.audio_x:
            ref = self._make_ref(key)
            if ref is not None:
                eg = dict()
                eg["mix"] = mix
                eg["ref"] = ref
                yield eg


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """

    def __init__(self, chunk_size, least=16000):
        self.chunk_size = chunk_size
        self.least = least

    def _make_chunk(self, eg, s):
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[s:s + self.chunk_size] for ref in eg["ref"]]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return None
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
            # random select start point
            s = random.randint(0, N % self.least)
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


def round_robin(objs, number):
    """
    Split list of objects
    Return list of object list
    """

    # at least one
    size = len(objs) // number
    if not size:
        return []

    obj_lists = [[] for _ in range(size)]
    idx = 0
    for obj in objs[:size * number]:
        obj_lists[idx].append(obj)
        idx = (idx + 1) % size
    return obj_lists


class DataLoader(object):
    def __init__(self,
                 perutt_loader,
                 chunk_size=32000,
                 batch_size=16,
                 cache_size=256):
        self.loader = perutt_loader
        self.cache_size, self.batch_size = cache_size, batch_size
        self.splitter = ChunkSplitter(chunk_size)

    def make_batch(self, egs, force=False):
        def T(ndarray):
            return th.tensor(ndarray, dtype=th.float32)

        if not force and len(egs) < self.cache_size:
            return None
        if not len(egs):
            return None

        eg_lists = round_robin(egs, self.batch_size)
        batch_list = []
        for eg_list in eg_lists:
            batch = dict()
            batch["mix"] = th.stack([T(eg["mix"]) for eg in eg_list])
            num_spks = len(eg_list[0]["ref"])
            batch["ref"] = [
                th.stack([T(eg["ref"][s]) for eg in eg_list])
                for s in range(num_spks)
            ]
            batch_list.append(batch)
        return batch_list

    def __iter__(self):
        egs = []
        for eg in self.loader:
            chunks = self.splitter.split(eg)
            if chunks is not None:
                egs.extend(chunks)
            elists = self.make_batch(egs)
            if elists is not None:
                yield from elists
                # keep remain
                r = len(egs) % self.batch_size
                # note if r == 0
                egs = egs[-r:] if r else []
        elists = self.make_batch(egs, force=True)
        if elists is not None:
            yield from elists
