import os
import csv

import numpy as np

def strip_series(input: str, max_chord_stack: int) -> (list, list):
    """
    Given some input string of sequential notes, parse it to produce
    two lists, separating the pitch from rhythm
    """
    pitch = list()
    rhythm = list()

    sequence = input.split(' + ')  # Each element is a series of simultaneous tokens
    for elem in sequence:
        tokens = elem.split(' ')

        # If super massive chord, then take random amount of them up to threshold
        if len(tokens) >= max_chord_stack:
            np.random.shuffle(tokens)
            tokens = tokens[:max_chord_stack]

        for t in tokens[:max_chord_stack]:

            if 'note' in t:
                pitch.append(t.split('_')[0])
                rhythm.append('note-' + t.split('_')[1])
            elif t != '\n':
                pitch.append(t)
                rhythm.append(t)

    return pitch, rhythm


def make_csv(data_cfg: dict) -> None:
    """
    Given an input directory and output path, for each .png and .semantic pair
    within the directory, write a row in an output csv of the .png path, and
    note pitch+rhythm of the .semantic file
    """
    data_dir = data_cfg["data_dir"]
    csv_out = data_cfg["csv_out"]
    max_chord_stack = data_cfg["max_chord_stack"]
    max_chord_stack = data_cfg["max_chord_stack"]

    # Get lists of valid tokens
    with open(data_cfg["vocab_pitch_path"], 'r') as f:
        pitch_tokens = f.read().split()

    with open(data_cfg["vocab_length_path"], 'r') as f:
        rhythm_tokens = f.read().split()

    with open(csv_out, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['img_path', 'pitch', 'rhythm'])

        data_imgs = [x for x in os.listdir(data_dir) if '.png' in x]
        for img in data_imgs:

            # Get respective .semantic file
            id = img.split('-')[0]
            num = img.split('-')[1].split('.')[0].lstrip('0')
            sem_file = id + '-' + num + '.semantic'
            if not os.path.isfile(os.path.join(data_dir, sem_file)):
                print(sem_file)
                break

            with open(os.path.join(data_dir, sem_file), 'r') as f_sem:
                contents = f_sem.read()

            pitch, rhythm = strip_series(contents, max_chord_stack)

            # Check all tokens are valid
            if len(set(pitch) - set(pitch_tokens)) > 0:
                continue
            if len(set(rhythm) - set(rhythm_tokens)) > 0:
                continue

            writer.writerow([img, pitch, rhythm])