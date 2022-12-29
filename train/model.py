import torch
import torch.nn as nn


class BaselineModel(torch.nn.Module):

    def __init__(self, model_cfg: dict, device: str, max_chord_stack: int, num_notes: int, num_lengths: int, img_height: int):
        super(BaselineModel, self).__init__()

        self.device = device
        self.img_height = img_height

        self.width_reduction = 1
        self.height_reduction = 1
        self.model_cfg = model_cfg
        self.max_chord_stack = max_chord_stack  # Largest possible chord expected

        # Calculate width and height reduction
        for i in range(4):
            self.width_reduction = self.width_reduction * model_cfg['conv_pooling_size'][i][1]
            self.height_reduction = self.height_reduction * model_cfg['conv_pooling_size'][i][0]

        # Conv blocks (4)
        self.b1 = nn.Sequential(
            nn.Conv2d(model_cfg['img_channels'], model_cfg['conv_filter_n'][0], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_cfg['conv_filter_n'][0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=model_cfg['conv_pooling_size'][0], stride=model_cfg['conv_pooling_size'][0])
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(model_cfg['conv_filter_n'][0], model_cfg['conv_filter_n'][1], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_cfg['conv_filter_n'][1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(model_cfg['conv_filter_n'][1], model_cfg['conv_filter_n'][2], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_cfg['conv_filter_n'][2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(model_cfg['conv_filter_n'][2], model_cfg['conv_filter_n'][3], kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(model_cfg['conv_filter_n'][3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )


        # Recurrent block
        rnn_hidden_units = self.model_cfg['rnn_units']
        rnn_hidden_layers = self.model_cfg['rnn_layers']
        feature_dim = self.model_cfg['conv_filter_n'][-1] * (self.img_height / self.height_reduction)

        # Bidirectional RNN
        self.r1 = nn.LSTM(int(feature_dim), hidden_size=rnn_hidden_units, num_layers=rnn_hidden_layers, dropout=0.5, bidirectional=True)

        # Split embedding parameters
        self.num_notes = num_notes
        self.num_lengths = num_lengths

        # Split embedding layers
        self.note_emb = nn.Linear(2 * rnn_hidden_units, self.num_notes + 1)     # +1 for blank symbol
        self.length_emb = nn.Linear(2 * rnn_hidden_units, self.num_lengths + 1) # +1 for blank symbol

        # Log Softmax at end for CTC Loss (dim = vocab dimension)
        self.sm = nn.LogSoftmax(dim=2)

        print('Vocab size:', num_lengths + num_notes)

    def forward(self, x):
        width_reduction = self.width_reduction
        height_reduction = self.height_reduction
        input_shape = x.shape  # (batch, channels, height, width)
        
        # Conv blocks (4)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        
        # Prepare output of conv block for recurrent blocks
        features = x.permute(3, 0, 2, 1)  # (width, batch, height, channels)
        feature_dim = self.model_cfg['conv_filter_n'][-1] * (self.img_height // height_reduction)
        feature_width = (2*2*2*input_shape[3]) // (width_reduction)
        stack = (int(feature_width), input_shape[0], int(feature_dim))
        features = torch.reshape(features, stack)  # -> [width, batch, features]

        # Recurrent block
        rnn_out, _ = self.r1(features)

        # Split embeddings
        note_out = self.note_emb(rnn_out)
        length_out = self.length_emb(rnn_out)

        # Log softmax (for CTC Loss)
        note_logits = self.sm(note_out)
        length_logits = self.sm(length_out)

        return note_logits, length_logits
