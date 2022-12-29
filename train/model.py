import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, model_cfg: dict, device: str, img_height: int):
        super(EncoderRNN, self).__init__()

        self.device = device
        self.img_height = img_height

        self.width_reduction = 1
        self.height_reduction = 1
        self.model_cfg = model_cfg

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
        rnn_hidden_units = self.model_cfg['encoder_rnn_units']
        rnn_hidden_layers = self.model_cfg['encoder_rnn_layers']
        feature_dim = self.model_cfg['conv_filter_n'][-1] * (self.img_height / self.height_reduction)

        # Bidirectional RNN
        self.r1 = nn.RNN(int(feature_dim), hidden_size=rnn_hidden_units, num_layers=rnn_hidden_layers, dropout=0.5, bidirectional=False)

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
        features = torch.reshape(features, stack)  # (width, batch, features)

        # Recurrent block
        output, hidden = self.r1(features)  # (width, batch, D*rnn_hidden_units), (D*rnn_hidden_layers, batch, rnn_hidden_units)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, model_cfg: dict, device: str, vocab_size: int):
        super(DecoderRNN, self).__init__()
        self.model_cfg = model_cfg

        self.hidden_size = self.model_cfg['decoder_hidden_size']
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size+1, self.hidden_size)
        self.ReLU = nn.ReLU()

        self.r1 = nn.RNN(self.hidden_size, hidden_size=self.hidden_size, num_layers=self.model_cfg['encoder_rnn_layers'], dropout=0.5, bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.vocab_size+1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, encoder_hidden):
        output = self.embedding(input)
        output = self.ReLU(output)
        output = torch.reshape(output, (output.shape[1], output.shape[0], -1))

        _, decoder_hidden = self.r1(output, encoder_hidden)
        last_hidden = decoder_hidden[decoder_hidden.shape[0] - 1]

        logits = self.out(last_hidden)
        return logits
