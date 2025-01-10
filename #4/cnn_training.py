#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
from torch import cuda, no_grad

import torch.nn as nn
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.optim import Adam

from cnn_system import MyCNNSystem
# .........
from pathlib import Path
from data_loader import load_data

# ............

__docformat__ = 'reStructuredText'


def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    epochs = 100

    # Instantiate our DNN
    # ..................
    # Define the CNN model and give it the model hyperparameters
    cnn_model = MyCNNSystem(

        cnn_channels_out_1=32,  # cnn_channels_1 # start with 32
        cnn_kernel_1=3,  ###
        cnn_stride_1=2,
        cnn_padding_1=2,
        pooling_kernel_1=3,  ###
        pooling_stride_1=2,
        cnn_channels_out_2=128,  # cnn_channels_1
        cnn_kernel_2=5,  ###
        cnn_stride_2=2,
        cnn_padding_2=2,
        pooling_kernel_2=5,  ###
        pooling_stride_2=2,
        classifier_input_features=4992,
        output_classes=1,
        dropout=0.25)

    # .................

    # Pass DNN to the available device.
    cnn_model = cnn_model.to(device)

    # Define the optimizer and give the parameters of the CNN model to an optimizer.
    optimizer = Adam(params=cnn_model.parameters(), lr=0.001)

    # Instantiate the loss function as a class.
    loss_function = BCEWithLogitsLoss()

    # Init training, validation, and testing dataset.
    data_path = Path('music_speech_dataset')
    batch_size = 4  # burası iyi 8 gibi olabilir, small dataset

    # ............
    split = "training"
    train_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)

    split = 'validation'
    valid_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)

    split = 'testing'
    test_loader = load_data(data_path, split, batch_size, shuffle=False, drop_last=True, num_workers=1)

    # Variables for the early stopping
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

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        cnn_model.train()

        # For each batch of our dataset.
        for batch in train_loader:
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, y = batch

            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device).float()

            # Get the predictions .
            y_hat = cnn_model(x).squeeze(1)

            # Calculate the loss
            # output = loss(m(input), target)
            # print("y_hat", y_hat.shape)
            # print("y", y.shape)
            # print("y_hat", y_hat.dtype)
            # print("y", y.unsqueeze(1).dtype )
            loss = loss_function(y_hat, y)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Append the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode
        cnn_model.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():

            # For every batch of our validation data.
            for batch in valid_loader:
                # Get the batch
                x_val, y_val = batch

                # Pass the data to the appropriate device.
                x_val = x_val.to(device)
                y_val = y_val.to(device).float()

                # Get the predictions of the model.
                y_hat = cnn_model(x_val).squeeze(1)

                # Calculate the loss.
                loss = loss_function(y_hat, y_val)

                # Append the validation loss.
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(cnn_model.state_dict())
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
                cnn_model.eval()
                with no_grad():
                    for batch in test_loader:
                        # x_test, y_test = ? 
                        x_test, y_test = batch
                        # Pass the data to the appropriate device.
                        # x_test = ?
                        # y_test = ?
                        x_test = x_test.to(device)
                        y_test = y_test.to(device).float()
                        # make the prediction
                        # y_hat = ?
                        y_hat = cnn_model(x_test).squeeze(1)

                        # Calculate the loss.
                        # loss = ?
                        loss = loss_function(y_hat, y_test)

                        testing_loss.append(loss.item())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')
                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()

# OBSERVATIONS: Firstly, I prefer to give a small value to batch_size since our data is very small. When it comes to
# strides which denotes the number of pixels shifts over the input, again I did not give big value since I don't want
# to downsample my image that much. As for kernel size, I believe 3x3 or 5x5 is suitable for this task, hence I use
# those either in pooling or convolutional part. And for the padding, I prefer to give value of 2 since it allows for
# a more accurate analysis of images. As for the outputs, it finds the model with the lowest mean loss in validation
# data and then using that model, we make the testing with testing data. For the hyperparameter values above,
# the model with the lowest loss has very low percentage, but the testing loss came a bit higher.


# EOF
