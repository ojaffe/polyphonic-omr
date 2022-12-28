import cv2
import random
import yaml

import numpy as np

import torch


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    path: path to YAML configuration file
    return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    return cfg


def normalize(image):
    """
    Makes black pixels of image take on high value, white pixels
    take on low value
    """
    return (255. - image)/255.


def resize(image, height, width=None):
    """
    Resizes an image to desired width and height
    """

    # Have width be original width
    if width is None:
        width = int(float(height * image.shape[1]) / image.shape[0])

    sample_img = cv2.resize(image, (width, height))
    return sample_img


def greedy_decode(logits, lengths):
    """
    logits = (seq len, batch size, vocab size)
    lengths = (batch size list where length of corresponding seq)
    return = (batch size list of lists where greedily decoded)
    """

    predictions = []
    blank_val = int(logits.shape[2]) - 1

    for batch_idx in range(logits.shape[1]):
        seq = []
        for seq_idx in range(lengths[batch_idx].item()):
            seq.append(int(logits[seq_idx][batch_idx].argmax().item()))
        
        new_seq = []
        prev = -1
        for s in seq:
            # Skip blanks and repeated
            if s == blank_val:
                prev = -1
                continue
            elif s == prev:
                continue

            new_seq.append(s)
            prev = s
        
        predictions.append(new_seq)

    return predictions
