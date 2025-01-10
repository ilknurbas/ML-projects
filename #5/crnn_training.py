#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sed_data_loading import split_data
from crnn_system import MyCRNNSystem
import pathlib
from copy import deepcopy

from torch import cuda, no_grad
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

import numpy as np


def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    crnn_model = MyCRNNSystem()

    crnn_model.to(device)

    optimizer = Adam(params=crnn_model.parameters(), lr=1e-3)

    loss_function = BCEWithLogitsLoss()

    data_path = pathlib.Path('sed_dataset')

    batch_size = 8

    train_loader, validation_loader, test_loader = split_data(data_path, batch_size)

    # Variables for the early stopping
    epochs = 300
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 30
    patience_counter = 0

    best_model = None

    # Start training.
    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode
        crnn_model.train()

        for batch in train_loader:
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y = batch

            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device)

            # Get the predictions of our model.
            # y_hat = crnn_model(x).squeeze(1) #type_as
            y_hat = crnn_model(x)

            # Calculate the loss of our model.
            loss = loss_function(input=y_hat, target=y.type_as(y_hat))

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Loss the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode
        crnn_model.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():

            # For every batch of our validation data.
            for batch in validation_loader:
                # Get the batch
                x_val, y_val = batch
                # Pass the data to the appropriate device.
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                # Get the predictions of the model.
                y_hat = crnn_model(x_val).squeeze(1)

                # Calculate the loss.
                # loss = loss_function(y_hat, y_val)
                loss = loss_function(input=y_hat, target=y_val.type_as(y_hat))

                # Log the validation loss.
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(crnn_model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if patience_counter >= patience:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                crnn_model.eval()
                with no_grad():
                    for batch in test_loader:
                        x_test, y_test = batch
                        x_test = x_test.to(device)
                        y_test = y_test.to(device)

                        # y_hat = crnn_model(x_test).squeeze(1)
                        y_hat = crnn_model(x_test)

                        # loss = loss_function(y_hat, y_test)
                        loss = loss_function(input=y_hat, target=y_test.type_as(y_hat))

                        testing_loss.append(loss.item())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')
                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()

# EOF
