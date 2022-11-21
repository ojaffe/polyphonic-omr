import argparse
import os
    
from utils import set_seed, load_config, greedy_decode
from dataset import load_data
from model import RNNDecoder

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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

    writer = SummaryWriter(log_dir=cfg["training"].get("log_dir", './log/'))

    # Set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_loader, val_loader, test_loader, blank_val_note, blank_val_length = load_data(data_cfg=cfg["data"])

    max_chord_stack = cfg["data"].get("max_chord_stack", 10)
    model = RNNDecoder(cfg["model"], device, max_chord_stack, blank_val_note, blank_val_length, cfg["data"].get("img_height", 128))
    model.to(device)

    # Optimisation
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"].get("lr", 1e-4)))
    pitch_loss_ctc = torch.nn.CTCLoss(blank=blank_val_note, zero_infinity=True)
    rhythm_loss_ctc = torch.nn.CTCLoss(blank=blank_val_length, zero_infinity=True)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
    model.apply(init_weights)

    # Training loop
    batch_size = cfg["data"].get("batch_size")
    max_epochs = cfg["training"].get("max_epochs", 500)

    print_training = cfg["training"].get("print_training")
    print_every = cfg["training"].get("print_every")
    decode_every = cfg["training"].get("decode_every")
    save_every = cfg["training"].get("save_every")

    for epoch in range(max_epochs):
        print("Epoch {:4d}".format(epoch))

        train_loss = 0

        # Train
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            image, pitch_seq, rhythm_seq = batch
            
            optimizer.zero_grad()
            pitch_pred, rhythm_pred = model(image.to(device))

            target_lengths = torch.tensor([len(x) for x in pitch_seq])
            pred_lengths = torch.tensor(pitch_pred.shape[0]).repeat(1, pitch_pred.shape[1]).squeeze(0)

            # Calculate CTC loss
            pitch_loss = pitch_loss_ctc(pitch_pred, pitch_seq, pred_lengths, target_lengths)
            rhythm_loss = rhythm_loss_ctc(rhythm_pred, rhythm_seq, pred_lengths, target_lengths)
            loss = pitch_loss + rhythm_loss

            loss.backward()   
            optimizer.step()
            train_loss += loss.item()
            
            # Tensorboard
            global_step = epoch*len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_pitch', pitch_loss, global_step)
            writer.add_scalar('Loss/train_rhythm', rhythm_loss, global_step)
            writer.add_scalar('Loss/train', loss, global_step)

            if (batch_idx+1) % print_every == 0 and print_training:
                print("Avg Loss: {:4.2f}".format(
                    train_loss / print_every
                ))
                train_loss = 0

            if (batch_idx+1) % decode_every == 0:
                greedy_preds_pitch = greedy_decode(rhythm_pred, pred_lengths)
                print(greedy_preds_pitch)
                print(rhythm_seq)

            if (batch_idx+1) % save_every == 0:
                save_model()   
                model_num += 1


        # Val
        model.eval()
        val_loss = 0
        for batch in val_loader:
            image, pitch_seq, rhythm_seq = batch
            
            with torch.no_grad():
                pitch_pred, rhythm_pred = model(image.to(device))

            target_lengths = torch.tensor([len(x) for x in pitch_seq])
            pred_lengths = torch.tensor(pitch_pred.shape[0]).repeat(1, pitch_pred.shape[1]).squeeze(0)

            # Calculate CTC loss
            pitch_loss = pitch_loss_ctc(pitch_pred, pitch_seq, target_lengths, target_lengths)
            rhythm_loss = rhythm_loss_ctc(rhythm_pred, rhythm_seq, target_lengths, target_lengths)
            loss = pitch_loss + rhythm_loss

            val_loss += loss

        print("Val loss: {:4.2f}".format(val_loss))
        writer.add_scalar('Loss/val', val_loss, global_step)


    writer.close()


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
