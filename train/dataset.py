import os
import cv2
import ast

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

from make_csv import make_csv
from utils import resize


class PolyphonicDataset(Dataset):
    def __init__(self, data_cfg: dict, note2idx: dict, idx2note: dict, vocab_size: int):

        self.data = pd.read_csv(data_cfg.get("csv_out", None))
        self.data_len = len(self.data)

        self.data_cfg = data_cfg

        self.note2idx = note2idx
        self.idx2note = idx2note
        self.vocab_size = vocab_size

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path, notes = self.data.iloc[idx]

        notes = ast.literal_eval(notes)

        # Deal with alpha (transparent PNG) - POLYPHONIC DATASET IMAGES
        sample_img = cv2.imread(os.path.join(self.data_cfg.get("data_dir", None), img_path), cv2.IMREAD_UNCHANGED)
        try:
            if sample_img.shape[2] == 4:     # we have an alpha channel
                a1 = ~sample_img[:,:,3]        # extract and invert that alpha
                sample_img = cv2.add(cv2.merge([a1,a1,a1,a1]), sample_img)   # add up values (with clipping)
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGBA2RGB)    # strip alpha channel
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY)
            elif sample_img.shape[2] == 3:   # no alpha channel (musicma_abaro)
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2GRAY) 
        except IndexError: # 2d image
            pass

        height = self.data_cfg.get("img_height", 128)
        sample_img = resize(sample_img, height, 880)

        tokens = torch.tensor([self.note2idx[x] for x in notes])

        return torch.tensor(sample_img, dtype=torch.float32).unsqueeze(0), tokens


class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs, tokens = zip(*batch)

        imgs = torch.stack(list(imgs), dim=0)
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.pad_idx)

        return imgs, tokens


def load_data(data_cfg: dict) -> (PolyphonicDataset, PolyphonicDataset, PolyphonicDataset, int, int):

    # Create vocab dict
    with open(data_cfg.get("vocab_path", None), 'r') as f:
        words = f.read().split('\n')
        note2idx = dict()
        idx2note = dict()
        for i, word in enumerate(words):
            note2idx[word] = i
            idx2note[i] = word
        vocab_size = len(note2idx)

    batch_size = data_cfg.get("batch_size", 4)

    # Create dataset from cleaned data, if doesn't already exist
    csv_out = data_cfg.get("csv_out", None)
    if not os.path.isfile(csv_out) or data_cfg.get("remake_csv", False):
        make_csv(data_cfg)

    dataset = PolyphonicDataset(data_cfg, note2idx, idx2note, vocab_size)

    # Create splits
    indices = list(range(len(dataset)))
    if data_cfg.get("shuffle", True):
        np.random.shuffle(indices)

    train_prop, val_prop, test_prop = data_cfg.get("dataset_split", [.8, .1, .1])
    train_split = int(np.floor(train_prop * len(dataset)))
    val_split = train_split + int(np.floor(val_prop * len(dataset)))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), 
                                                    collate_fn=PadCollate(dataset.vocab_size))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), 
                                                    collate_fn=PadCollate(dataset.vocab_size))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), 
                                                    collate_fn=PadCollate(dataset.vocab_size))

    return train_loader, val_loader, test_loader, note2idx, idx2note, vocab_size