import argparse
import os
from itertools import zip_longest
    
from utils import set_seed, load_config, greedy_decode
from dataset import load_data
from model import EncoderRNN, DecoderRNN

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

    cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    writer = SummaryWriter(log_dir=cfg["training"].get("log_dir", './log/'))

    # Set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_loader, val_loader, test_loader, note2idx, idx2note, vocab_size = load_data(data_cfg=cfg["data"])

    max_chord_stack = cfg["data"].get("max_chord_stack", 10)
    encoder = EncoderRNN(cfg["model"], device, cfg["data"].get("img_height", 128)).to(device)
    decoder = DecoderRNN(cfg["model"], device, vocab_size).to(device)

    # Optimisation
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=float(cfg["training"].get("lr", 1e-4)))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(cfg["training"].get("lr", 1e-4)))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab_size)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
    encoder.apply(init_weights)
    decoder.apply(init_weights)

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
        encoder.train()
        decoder.train()
        for batch_idx, batch in enumerate(train_loader):
            image, tokens = batch
            image = image.to(device)
            tokens = tokens.to(device)
            
            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            _, encoder_hidden = encoder(image)

            # Teacher forcing
            all_logits = torch.tensor(list()).to(device)
            target_length = len(tokens[0])
            for interval in range(target_length):
                tokens_sub = tokens[:, :interval+1]
                logits = decoder(tokens_sub, encoder_hidden)
                all_logits = torch.cat((all_logits, torch.reshape(logits, (logits.shape[0], 1, -1))), dim=1)

                # Get logits of
                loss += criterion(logits, torch.tensor([x[interval] for x in tokens]).to(device))

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Tensorboard
            global_step = epoch*len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss, global_step)

            if (batch_idx+1) % decode_every == 0:
                pred_tokens = greedy_decode(all_logits[0], note2idx, idx2note)
                preds_and_actual = list(zip_longest(pred_tokens, tokens[0].tolist(), fillvalue=-1))
                
                writer.add_text('Train pred', ' '.join([str(x) for x in preds_and_actual]), global_step)

        # Val
        val_loss = 0
        encoder.eval()
        decoder.eval()
        for batch_idx, batch in enumerate(val_loader):
            image, tokens = batch
            image = image.to(device)
            tokens = tokens.to(device)
            
            
            with torch.no_grad():
                _, encoder_hidden = encoder(image)

                # Teacher forcing
                all_logits = torch.tensor(list()).to(device)
                target_length = len(tokens[0])
                for interval in range(target_length):
                    tokens_sub = tokens[:, :interval+1]
                    logits = decoder(tokens_sub, encoder_hidden)
                    all_logits = torch.cat((all_logits, torch.reshape(logits, (logits.shape[0], 1, -1))), dim=1)

                    # Get logits of
                    val_loss += criterion(logits, torch.tensor([x[interval] for x in tokens]).to(device))

        val_loss /= len(val_loader)

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
