#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Dropout2d, GRU, Linear, Conv2d, MaxPool2d, BatchNorm2d, ReLU


class MyCRNNSystem(Module):
    def __init__(self) -> None:
        super(MyCRNNSystem, self).__init__()

        self.cnn_block_1 = Sequential(
            # time should be 1024
            # features should be 40
            Conv2d(in_channels=1,
                   out_channels=128,
                   kernel_size=(1, 2), # 5-6
                   stride=(1, 2), # 1-2
                   padding=(0, 1)),
            ReLU(),
            BatchNorm2d(128),
            MaxPool2d(kernel_size=1,
                      stride=1),
            Dropout2d(0.5)
        )

        self.cnn_block_2 = Sequential(
            Conv2d(in_channels=128,
                   out_channels=8,
                   kernel_size=(1, 1),
                   stride=(1, 1),
                   padding=(0, 1)),
            ReLU(),
            BatchNorm2d(8),
            MaxPool2d(kernel_size=(1, 1),
                      stride=(1, 1)),
            Dropout2d(0.5))

        self.rnn_layer = GRU(input_size=184,
                             hidden_size=4,
                             num_layers=2,
                             batch_first=True)

        self.linear = Linear(in_features=4, out_features=6)

    def forward(self,
                X: Tensor) -> Tensor:
        X = X.float()
        X = X if X.ndimension() == 4 else X.unsqueeze(1)
        # print statements can be uncommented to check dimensionality
        # apply cnn_block_1 to X
        cnn_out_1 = self.cnn_block_1(X)
        #print("cnn_out_1", cnn_out_1.size())

        # apply cnn_block_2 to cnn_out_1
        cnn_out_2 = self.cnn_block_2(cnn_out_1)
        #print("cnn_out_2", cnn_out_2.size())

        # apply permute 
        cnn_out_2 = cnn_out_2.permute(0, 2, 3, 1)
        #print("after permute", cnn_out_2.size())
        # reshape
        cnn_out_2 = cnn_out_2.reshape(cnn_out_2.shape[0], cnn_out_2.shape[1], -1) #(0, 2, cnn_out_2.size()[2]*cnn_out_2.size()[3]) ####
        #print("after reshape", cnn_out_2.size())

        # apply rnn_layer
        rnn_out, _ = self.rnn_layer(cnn_out_2)
        #print("rnn_out", rnn_out.size())
        # apply linear layer
        y_hat = self.linear(rnn_out)
        #print("linear", y_hat.size())

        return y_hat


# EOF
