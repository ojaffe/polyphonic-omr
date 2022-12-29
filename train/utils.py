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


def greedy_decode(logits, note2idx, idx2note):
    """
    logits = (seq len, batch size, vocab size)
    return = (batch size list of lists where greedily decoded)
    """

    eos_idx = note2idx['<EOS>']

    predictions = list()
    for i in range(logits.shape[0]):
        idx = torch.argmax(logits[i])
        predictions.append(idx.item())
        if idx == eos_idx:
            break

    return predictions
