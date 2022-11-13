import os
import cv2

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

from make_csv import make_csv
from utils import resize


class PolyphonicDataset(Dataset):
    def __init__(self, data_cfg: dict):

        self.data = pd.read_csv(data_cfg.get("csv_out", None), converters={'pitch': pd.eval, 'rhythm': pd.eval})
        self.data_len = len(self.data)

        self.data_cfg = data_cfg
        vocab_pitch_path = data_cfg.get("vocab_pitch_path", None)
        vocab_length_path = data_cfg.get("vocab_length_path", None)

        # Load in dictionary (either note or combined one)
        with open(vocab_pitch_path, 'r') as f:
            words = f.read().split()
            self.note2idx = dict()
            self.idx2note = dict()
            for i, word in enumerate(words):
                self.note2idx[word] = i
                self.idx2note[i] = word
            self.vocab_size_note = len(self.note2idx)

        # Load in length dictionary if being used 
        with open(vocab_length_path, 'r') as f:
            words = f.read().split()
            self.length2idx = dict()
            self.idx2length = dict()
            for i, word in enumerate(words):
                self.length2idx[word] = i
                self.idx2length[i] = word
            self.vocab_size_length = len(self.length2idx)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path, pitch_seq, rhythm_seq = self.data.iloc[idx]

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
        sample_img = resize(sample_img,height, int(float(height * sample_img.shape[1]) / sample_img.shape[0]) // 8 * 8)

        pitch_idxs = torch.tensor([self.note2idx[x] for x in pitch_seq])
        rhythm_idxs = torch.tensor([self.length2idx[x] for x in rhythm_seq])

        return torch.tensor(sample_img, dtype=torch.float32).unsqueeze(0), pitch_idxs, rhythm_idxs


class PadCollate:
    def __init__(self, pitch_pad_idx, rhythm_pad_idx):
        self.pitch_pad_idx = pitch_pad_idx
        self.rhythm_pad_idx = rhythm_pad_idx

    def __call__(self, batch):
        imgs, pitchs, rhythms = zip(*batch)

        max_seq_length = max([len(x) for x in pitchs])

        pitchs = pad_sequence(pitchs, batch_first=True, padding_value=self.pitch_pad_idx)
        rhythms = pad_sequence(rhythms, batch_first=True, padding_value=self.rhythm_pad_idx)

        imgs = torch.stack(list(imgs), dim=0)

        return imgs, pitchs, rhythms


def load_data(data_cfg: dict) -> (PolyphonicDataset, PolyphonicDataset, PolyphonicDataset, int, int):
    batch_size = data_cfg.get("batch_size", 4)

    # Create dataset from cleaned data, if doesn't already exist
    csv_out = data_cfg.get("csv_out", None)
    if not os.path.isfile(csv_out):
        make_csv(data_cfg)

    dataset = PolyphonicDataset(data_cfg)

    # Create splits
    indices = list(range(len(dataset)))
    if data_cfg.get("shuffle", True):
        np.random.shuffle(indices)

    train_prop, val_prop, test_prop = data_cfg.get("dataset_split", [80, 10, 10])
    train_split = int(np.floor(train_prop * len(dataset)))
    val_split = train_split + int(np.floor(val_prop * len(dataset)))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[train_split+val_split:]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), 
                                                    collate_fn=PadCollate(dataset.vocab_size_note, dataset.vocab_size_length))
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), 
                                                    collate_fn=PadCollate(dataset.vocab_size_note, dataset.vocab_size_length))
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices), 
                                                    collate_fn=PadCollate(dataset.vocab_size_note, dataset.vocab_size_length))

    return train_loader, val_loader, test_loader, dataset.vocab_size_note, dataset.vocab_size_length