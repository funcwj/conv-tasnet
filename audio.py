# wujian@2018

from utils import read_wav


def parse_scps(scp_path, value_processor=lambda x: x, num_tokens=2):
    """
    Parse kaldi's script(.scp) file
    If num_tokens >= 2, function will check token number
    """
    scp_dict = dict()
    line = 0
    with open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if num_tokens >= 2 and len(scp_tokens) != num_tokens or len(
                    scp_tokens) < 2:
                raise RuntimeError(
                    "For {}, format error in line[{:d}]: {}".format(
                        scp_path, line, raw_line))
            if num_tokens == 2:
                key, value = scp_tokens
            else:
                key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError("Duplicated key \'{0}\' exists in {1}".format(
                    key, scp_path))
            scp_dict[key] = value_processor(value)
    return scp_dict


class Reader(object):
    """
        Basic Reader Class
    """

    def __init__(self, scp_path, value_processor=lambda x: x):
        self.index_dict = parse_scps(
            scp_path, value_processor=value_processor, num_tokens=2)
        self.index_keys = list(self.index_dict.keys())

    def _load(self, key):
        # return path
        return self.index_dict[key]

    # number of utterance
    def __len__(self):
        return len(self.index_dict)

    # avoid key error
    def __contains__(self, key):
        return key in self.index_dict

    # sequential index
    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)

    # random index, support str/int as index
    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError("Unsupported index type: {}".format(type(index)))
        if type(index) == int:
            # from int index to key
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError(
                    "Interger index out of range, {:d} vs {:d}".format(
                        index, num_utts))
            index = self.index_keys[index]
        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))
        return self._load(index)


class WaveReader(Reader):
    """
        Sequential/Random Reader for single channel wave
        Format of wav.scp follows Kaldi's definition:
            key1 /path/to/wav
            ...
    """

    def __init__(self, wav_scp, sample_rate=None, normalize=True):
        super(WaveReader, self).__init__(wav_scp)
        self.samp_rate = sample_rate
        self.normalize = normalize

    def _load(self, key):
        # return C x N or N
        samp_rate, samps = read_wav(
            self.index_dict[key], normalize=self.normalize, return_rate=True)
        # if given samp_rate, check it
        if self.samp_rate is not None and samp_rate != self.samp_rate:
            raise RuntimeError("SampleRate mismatch: {:d} vs {:d}".format(
                samp_rate, self.samp_rate))
        return samps