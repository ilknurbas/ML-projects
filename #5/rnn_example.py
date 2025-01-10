#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import chain

import torch
from torch import rand, Tensor
from torch.cuda import is_available
from torch.nn import GRU, LSTM, Linear, MSELoss, Sigmoid, BCEWithLogitsLoss
from torch.optim import Adam

__docformat__ = 'reStructuredText'
__all__ = []


def rnns_gru():
    device = 'cuda' if is_available() else 'cpu'

    epochs = 100
    batch_size = 8
    nb_batches = 10
    nb_examples = batch_size * nb_batches
    t_steps = 64
    in_features = 8
    out_features = 2

    # Create a two-layer GRU with batch_first=True
    rnn = GRU(input_size=in_features, hidden_size=4, num_layers=2, batch_first=True)
    rnn = rnn.to(device)
    # Create the Linear layer
    # in_features???
    linear = Linear(in_features=4, out_features=out_features)
    linear = linear.to(device)
    # Define non-linearity to be added at the end
    activation = Sigmoid()
    # Define the loss function as MSELoss
    loss_f = MSELoss()

    optimizer = Adam(chain(rnn.parameters(), linear.parameters()))

    # Create some dummy data for input and target values
    x = rand(nb_examples, t_steps, in_features)  # , dtype=torch.float64
    x = x.to(device)
    y = rand(nb_examples, t_steps, out_features)  # , dtype=torch.float64
    y = y.to(device)

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(0, nb_examples, batch_size):
            optimizer.zero_grad()
            x_in = x[i:i + batch_size, :, :]
            y_out = y[i:i + batch_size, :, :]
            y_hat = activation(linear(rnn(x_in)[0]))

            loss = loss_f(y_hat, y_out)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')


def rnns_lstm():
    device = 'cuda' if is_available() else 'cpu'

    epochs = 100
    batch_size = 8
    nb_batches = 10
    nb_examples = batch_size * nb_batches
    t_steps = 64
    in_features = 8
    out_features = 2

    # Create a two-layer LSTM with batch_first=True
    rnn = LSTM(input_size=in_features, hidden_size=4, num_layers=2, batch_first=True)
    rnn = rnn.to(device)
    # Create the Linear layer
    # in_features
    linear = Linear(in_features=4, out_features=out_features)
    linear = linear.to(device)
    # Define non-linearity to be added at the end
    activation = Sigmoid()
    # Define the loss function as MSELoss
    loss_f = MSELoss()

    optimizer = Adam(chain(rnn.parameters(), linear.parameters()))

    # Create some dummy data for input and target values
    x = rand(nb_examples, t_steps, in_features)  # , dtype=torch.float64
    x = x.to(device)
    y = rand(nb_examples, t_steps, out_features)  # , dtype=torch.float64
    y = y.to(device)

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(0, nb_examples, batch_size):
            optimizer.zero_grad()
            x_in = x[i:i + batch_size, :, :]
            y_out = y[i:i + batch_size, :, :]
            y_hat = activation(linear(rnn(x_in)[0]))

            loss = loss_f(y_hat, y_out)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')


def k_hot_encoding_case():
    device = 'cuda' if is_available() else 'cpu'

    epochs = 100
    batch_size = 8
    nb_batches = 10
    nb_examples = batch_size * nb_batches
    t_steps = 20
    in_features = 8
    out_features = 4

    # Create a two-layer GRU or LSTM with batch_first=True
    rnn = GRU(input_size=in_features, hidden_size=4, num_layers=2, batch_first=True)
    rnn = rnn.to(device)
    # Create the Linear layer
    linear = Linear(in_features=4, out_features=out_features)
    linear = linear.to(device)
    # This loss combines a Sigmoid layer and the BCELoss in one single class.
    loss_f = BCEWithLogitsLoss()

    optimizer = Adam(chain(rnn.parameters(), linear.parameters()))

    # Create some dummy data for input and target values
    x = rand(nb_examples, t_steps, in_features)  # , dtype=torch.float64
    x = x.to(device)
    # another implementation:
    # y = rand(nb_examples, t_steps, out_features).ge(0.5).float()
    y = torch.randint(low=0, high=2, size=(nb_examples, t_steps, out_features)).float()
    y = y.to(device)

    for epoch in range(epochs):
        epoch_loss = []

        for i in range(0, nb_examples, batch_size):
            optimizer.zero_grad()
            x_in = x[i:i + batch_size, :, :]
            y_out = y[i:i + batch_size, :, :]
            y_hat = linear(rnn(x_in)[0])

            loss = loss_f(y_hat, y_out)
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        print(f'Epoch: {epoch:03d} | Loss: {Tensor(epoch_loss).mean():7.4f}')


def main():
    print('Running GRU case')
    rnns_gru()
    print('-' * 100)
    print('Running LSTM case')
    rnns_lstm()
    print('-' * 100)
    print('Running k-hot encoding case')
    k_hot_encoding_case()


if __name__ == '__main__':
    main()

# EOF
