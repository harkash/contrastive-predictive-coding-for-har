import numpy as np
import os
import torch
import torch.nn as nn


class CPC(nn.Module):
    def __init__(self, args):
        super(CPC, self).__init__()

        # 1D conv Encoder to get the outputs at each timestep
        self.encoder = Encoder(args)

        # RNN to obtain context vector
        self.rnn = nn.GRU(input_size=128,
                          hidden_size=256,
                          num_layers=2,
                          bidirectional=False,
                          batch_first=True,
                          dropout=0.2)

        # Projections for k steps
        self.Wk = nn.ModuleList([PredictionNetwork()
                                 for i in range(args.num_steps_prediction)])

        # Softmaxes for the loss computation
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        # Other details
        self.batch_size = args.batch_size
        self.seq_len = args.window
        self.num_steps_prediction = args.num_steps_prediction
        self.device = args.device

    def forward(self, inputs):
        # Passing through the encoder. Input: BxCxT and output is: Bx128XT
        z = self.encoder(inputs)

        # Random timestep to start the future prediction from.
        # If the window is 50 timesteps and k=12, we pick a number from 0-37
        start = torch.randint(int(inputs.shape[1] - self.num_steps_prediction),
                              size=(1,)).long()

        # Need to pick the encoded data only until the starting timestep
        rnn_input = z[:, :start + 1, :]

        # Passing through the RNN
        r_out, (_) = self.rnn(rnn_input, None)

        accuracy, nce, correct_steps = self.compute_cpc_loss(z, r_out, start)

        return accuracy, nce, correct_steps

    def compute_cpc_loss(self, z, c, t):
        batch_size = z.shape[0]

        # The context vector is the last timestep from the RNN
        c_t = c[:, t, :].squeeze(1)

        # infer z_{t+k} for each step in the future: c_t*Wk, where 1 <= k <=
        # timestep
        pred = torch.stack([self.Wk[k](c_t) for k in
                            range(self.num_steps_prediction)])

        # pick the target z values k timestep number of samples after t
        z_samples = z[:, t + 1: t + 1 + self.num_steps_prediction, :] \
            .permute(1, 0, 2)

        nce = 0
        correct = 0
        correct_steps = []

        # Looping over the number of timesteps chosen
        for k in range(self.num_steps_prediction):
            # calculate the log density ratio: log(f_k) = z_{t+k}^T * W_k * c_t
            log_density_ratio = torch.mm(z_samples[k], pred[k].transpose(0, 1))

            # correct if highest probability is in the diagonal
            positive_batch_pred = torch.argmax(self.softmax(log_density_ratio),
                                               dim=0)
            positive_batch_actual = torch.arange(0, batch_size).to(self.device)
            correct = (correct +
                       torch.sum(torch.eq(positive_batch_pred,
                                          positive_batch_actual)).item())
            correct_steps.append(torch.sum(torch.eq(positive_batch_pred,
                                                    positive_batch_actual)).item())

            # calculate NCE loss
            nce = nce + torch.sum(torch.diag(self.lsoftmax(log_density_ratio)))

        # average over timestep and batch
        nce = nce / (-1.0 * batch_size * self.num_steps_prediction)
        accuracy = correct / (1.0 * batch_size * self.num_steps_prediction)
        correct_steps = np.array(correct_steps)
        return accuracy, nce, correct_steps

    def predict_features(self, inputs):
        z = self.encoder(inputs)

        # Passing through the RNN
        r_out, _ = self.rnn(z, None)

        return r_out


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.encoder = Convolutional1DEncoder(args)

    def forward(self, inputs):
        return self.encoder(inputs)


class Convolutional1DEncoder(nn.Module):
    def __init__(self, args):
        super(Convolutional1DEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(args.input_size, 32, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect'),
            ConvBlock(32, 64, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect'),
            ConvBlock(64, 128, kernel_size=args.kernel_size,
                      stride=1, padding=args.padding,
                      padding_mode='reflect')
        )

    def forward(self, inputs):
        # Tranposing since the Conv1D requires
        inputs = inputs.permute(0, 2, 1)
        encoder = self.encoder(inputs)
        encoder = encoder.permute(0, 2, 1)

        return encoder


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=1, padding_mode='reflect', dropout_prob=0.2):
        super(ConvBlock, self).__init__()

        # 1D convolutional layer
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        conv = self.conv(inputs)
        relu = self.relu(conv)
        dropout = self.dropout(relu)

        return dropout


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        # Encoder
        self.encoder = CPC(args)

        # Softmax
        self.softmax = nn.Sequential(nn.Linear(256, 256),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(256, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(128, args.num_classes))

    def forward(self, inputs):
        encoder = self.predict_features(inputs)

        softmax = self.softmax(encoder[:, -1, :])

        return softmax

    def predict_features(self, inputs):
        r_out = self.encoder.predict_features(inputs)

        return r_out

    def load_pretrained_weights(self, args):
        state_dict_path = os.path.join(args.saved_model)

        print('Loading the pre-trained weights')
        pretrained_checkpoint = torch.load(state_dict_path,
                                           map_location=args.device)

        # Appending encoder to the weight keys since that is how our
        # classifier model is setup, the first layer is encoder = which is
        # the CPC model
        updated_checkpoints = {}
        for k, v in pretrained_checkpoint.items():
            updated_checkpoints['encoder.' + k] = v

        self.load_state_dict(updated_checkpoints, False)

        return

    def freeze_encoder_layers(self):
        """
        To set only the softmax to be trainable
        :return: None, just setting the encoder part (or the CPC model) as
        frozen
        """
        # First setting the model to eval
        self.encoder.eval()

        # Then setting the requires_grad to False
        for param in self.encoder.parameters():
            param.requires_grad = False

        return


class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()
        self.Wk = nn.Linear(256, 128)

    def forward(self, inputs):
        prediction = self.Wk(inputs)

        return prediction
