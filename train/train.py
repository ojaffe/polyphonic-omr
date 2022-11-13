import argparse
import os
    
from utils import set_seed, load_config, greedy_decode
from dataset import load_data
from model import RNNDecoder

import torch
from torch.utils.data import DataLoader


# Function to save model
def save_model():
    # Save model
    root_model_path = 'models/latest_model' + str(model_num) + '.pt'
    model_dict = model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, root_model_path)
    print('Saved model')


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    # Set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_loader, val_loader, test_loader, blank_val_note, blank_val_length = load_data(data_cfg=cfg["data"])

    max_chord_stack = cfg["data"].get("max_chord_stack", 10)
    model = RNNDecoder(cfg["model"], device, max_chord_stack, blank_val_note, blank_val_length, cfg["data"].get("img_height", 128))
    model.to(device)

    # Optimisation
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"].get("lr", 1e-4)))
    pitch_loss = torch.nn.CTCLoss(blank=blank_val_note, zero_infinity=True)
    length_loss = torch.nn.CTCLoss(blank=blank_val_length, zero_infinity=True)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
    model.apply(init_weights)

    # Training loop
    max_epochs = cfg["training"].get("max_epochs", 500)
    for epoch in range(max_epochs):
        print('Epoch %d...' % epoch)

        train_loss = 0

        # Go through training data
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            image, pitch_seq, rhythm_seq = batch
            
            optimizer.zero_grad()
            pitch_pred, rhythm_pred = model(image.to(device))

            input_lengths = torch.tensor([len(x) for x in pitch_seq])

            # Calculate CTC loss
            loss = length_loss(rhythm_pred, rhythm_seq, input_lengths.clone().detach(), input_lengths.clone().detach()) + \
                pitch_loss(pitch_pred, pitch_seq, input_lengths.clone().detach(), input_lengths.clone().detach())

            loss.backward()   
            optimizer.step()
            train_loss += loss.item()

            if (batch_idx+1) % 40 == 0:
                greedy_preds_pitch = greedy_decode(pitch_pred, input_lengths)
                print(greedy_preds_pitch)
                print(pitch_seq)

            if (batch_idx+1) % 5 == 0:
                # Overall training loss
                if batch_idx == 0:
                    print ('Training loss value at batch %d: %f' % ((batch_idx),train_loss))
                else:
                    print ('Training loss value at batch %d: %f' % ((batch_idx),train_loss/500))
                train_loss = 0 

            if (batch_idx+1) % 1500 == 0:
                save_model()   
                model_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AHAHAAH")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)
