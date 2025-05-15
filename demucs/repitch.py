# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch
import librosa
import pytsmod as tsm

def i16_pcm(wav):
    if wav.dtype == np.int16:
        return wav
    return (wav * 2**15).clamp_(-2**15, 2**15 - 1).short()


def f32_pcm(wav):
    if wav.dtype == np.float32:
        return wav
    return wav.float() / 2**15


class RepitchedWrapper:
    """
    Wrap a dataset to apply online change of pitch / tempo.
    """
    def __init__(self, dataset, proba=0.5, max_pitch=3, max_tempo=25, mode="resample"):
        self.dataset = dataset
        self.proba = proba
        self.max_pitch = max_pitch
        self.max_tempo = max_tempo
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        streams = self.dataset[index]
        in_length = streams.shape[-1]
        out_length = int((1 - 0.01 * self.max_tempo) * in_length)
        
        if random.random() < self.proba:
            delta_pitch = random.randint(-self.max_pitch, self.max_pitch)
            delta_tempo = random.randint(-self.max_tempo, self.max_tempo)
            stretch_factor = 1 + delta_tempo / 100
            outs = []
            for idx, stream in enumerate(streams):
                x = stream.numpy()
                if self.mode == "resample":
                    x = librosa.resample(x, orig_sr=16000, target_sr=stretch_factor*16000)
                else:
                    x = librosa.effects.pitch_shift(x, sr=16000, n_steps=delta_pitch)
                    x = tsm.phase_vocoder(x, stretch_factor, phase_lock=True)      
                stream = torch.from_numpy(x.astype('float32'))
                if self.mode != "resample":
                    stream = stream.unsqueeze(0)
                outs.append(stream[..., :out_length])
            streams = torch.stack(outs)
        else:
            streams = streams[..., :out_length]
        return streams
