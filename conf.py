fs = 8000
chunk_len = 4  # (s)

# network configure
nnet_conf = {
    "L": 20,
    "N": 256,
    "X": 8,
    "R": 4,
    "B": 256,
    "H": 512,
    "P": 3,
    "norm": "BN",
    "num_spks": 2,
}

# data configure:
train_dir = "data/wsj0_2mix/tr/"
dev_dir = "data/wsj0_2mix/cv/"
chunk_size = chunk_len * fs

train_data = {
    "audio_x": train_dir + "mix.scp",
    "audio_y": [train_dir + "spk1.scp", train_dir + "spk2.scp"],
    "sample_rate": fs,
}

dev_data = {
    "audio_x": dev_dir + "mix.scp",
    "audio_y": [dev_dir + "spk1.scp", dev_dir + "spk2.scp"],
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 0,
    "factor": 0.5,
    "logging_period": 200  # batch number
}