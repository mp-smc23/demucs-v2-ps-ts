# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import torch as th
import tqdm
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from .parser import get_name, get_parser
from .utils import apply_model, average_metric, center_trim

def split_dataset(dataset, num_splits):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split_size = len(dataset) // num_splits
    subsets = [Subset(dataset, indices[i*split_size:(i+1)*split_size]) for i in range(num_splits)]
    return subsets


parser = get_parser()
args = parser.parse_args()

def train_model(epoch,
                dataset,
                model,
                criterion,
                optimizer,
                augment,
                quantizer=None,
                diffq=0,
                repeat=1,
                device="cpu",
                seed=None,
                workers=4,
                world_size=1,
                batch_size=16,
                clip_value=args.gradient_clip):
    
    #mll: minibatches: 
    subsets = split_dataset(dataset, num_splits=4)
    
    current_loss = 0
    model_size = 0
    
    batch_size //= world_size
    
    for subset_idx, subset in enumerate(subsets):
        if world_size > 1:
            sampler = DistributedSampler(subset)
            sampler_epoch = epoch * repeat
            if seed is not None:
                sampler_epoch += seed * 1000
            sampler.set_epoch(sampler_epoch)
            loader = DataLoader(subset, batch_size=batch_size, sampler=sampler, num_workers=workers)
        else:
            loader = DataLoader(subset, batch_size=batch_size, num_workers=workers, shuffle=True)
            
    for repetition in range(repeat):
        tq = tqdm.tqdm(loader,
                       ncols=120,
                       desc=f"[{epoch:03d}] train ({repetition + 1}/{repeat})",
                       leave=False,
                       file=sys.stdout,
                       unit=" batch")
        total_loss = 0
        for idx, sources in enumerate(tq):
            if len(sources) < batch_size:
                # skip uncomplete batch for augment.Remix to work properly
                continue
            sources = sources.to(device)
            sources = augment(sources)
            mix = sources.sum(dim=1)

            estimates = model(mix)
            sources = center_trim(sources, estimates)
            if sources.dim() == 4:
                # for 2D conv models
                estimates = estimates.squeeze(2)
                sources = sources.squeeze(2)
            loss = criterion(estimates, sources)
            model_size = 0
            if quantizer is not None:
                model_size = quantizer.model_size()

            train_loss = loss + diffq * model_size
            train_loss.backward()

            #added gradient clipping:
            clip_grad_norm_(model.parameters(), max_norm=clip_value)

            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm()**2
            grad_norm = grad_norm**0.5
            
            optimizer.step()
            optimizer.zero_grad()

            if quantizer is not None:
                model_size = model_size.item()

            total_loss += loss.item()
            current_loss = total_loss / (1 + idx)
            tq.set_postfix(loss=f"{current_loss:.4f}", ms=f"{model_size:.2f}",
                           grad=f"{grad_norm:.5f}")

            # free some space before next round
            del sources, mix, estimates, loss, train_loss

        if world_size > 1:
            sampler.epoch += 1

    if world_size > 1:
        current_loss = average_metric(current_loss)
    return current_loss, model_size


def validate_model(epoch,
                   dataset,
                   model,
                   criterion,
                   device="cpu",
                   rank=0,
                   world_size=1,
                   shifts=0,
                   overlap=0.25,
                   split=False):
    indexes = range(rank, len(dataset), world_size)

    tq = tqdm.tqdm(indexes,
                   ncols=120,
                   desc=f"[{epoch:03d}] valid",
                   leave=False,
                   file=sys.stdout,
                   unit=" track")
    
    current_loss = 0
    with th.no_grad():
        for index in tq:
            streams = dataset[index]
            # first five minutes to avoid OOM on --upsample models
            streams = streams[..., :15_000_000]
            streams = streams.to(device)
            sources = streams[1:]
            mix = streams[0]    
            estimates = apply_model(model, mix, shifts=shifts, split=split, overlap=overlap)
            loss = criterion(estimates, sources)
            current_loss += loss.item() / len(indexes)
            del estimates, streams, sources
            # print(th.cuda.memory_summary())
            # th.cuda.empty_cache()

    if world_size > 1:
        current_loss = average_metric(current_loss, len(indexes))
    return current_loss
