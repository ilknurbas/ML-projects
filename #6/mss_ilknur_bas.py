import librosa
import numpy as np
import torch

from dae_ilknur_bas import MySystem
from dataset_class import MyDataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import soundfile as sf
from exercise_06 import to_audio
from feature_extraction_ilknur_bas import create_sub_directories, save_serialized_stft
import mir_eval


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    model = MySystem()
    model = model.to(device)

    # Define hyper-parameters to be used.
    batch_size = 4
    epochs = 40  # 200
    learning_rate = 1e-4

    # create dataset
    create_sub_directories()  # create directory
    save_serialized_stft()  # save the serialized versions to respective folders

    # training dataset
    dataset_train = MyDataset(split=True, source_out_path="Dataset/Sources/training_features",
                              mix_in_path="Dataset/Mixtures/training_features")
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=False)
    print("data_loader_train", data_loader_train.__len__())

    # testing dataset
    dataset_test = MyDataset(split=False, mix_in="Dataset/Mixtures/testing_features",
                             # mix_raw_data="Dataset/Mixtures/testing/testing_1/mixture.wav",
                             mix_raw_data="Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav")
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)
    print("data_loader_test", data_loader_test.__len__())

    loss_function = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss_training = []

        model.train()
        for i, batch in enumerate(data_loader_train):
            optimizer.zero_grad()
            x, y = batch
            # Give them to the appropriate device.
            x = x.to(device)
            y = y.to(device)
            # print("mss_x.shape", x.shape)  # ([4, 1025, 60])
            # print("mss_y.shape", y.shape)  # ([4, 1025, 60])

            # Get the predictions of our model.
            y_hat = model(x)
            # print("mss_y_hat.shape", y_hat.shape)  # ([4, 1025, 60])

            # Calculate the loss of our model.
            loss = loss_function(input=y_hat, target=y.type_as(y_hat))

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Loss the loss of the batch
            epoch_loss_training.append(loss.item())

            # break;
        # break;

        print("* epoch {}".format(epoch))
        epoch_loss_training = np.array(epoch_loss_training).mean()
        print("Training loss ", epoch_loss_training)
        model.eval()
        # predicted_chunks_features = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader_test):
                x_test, y_test = batch
                x_test = x_test.to(device)
                y_test = y_test.to(device)  # not important in fact
                # print("mss_x_test.shape", x_test.shape)  # ([4, 60, 1025])
                # print("mss_y_test.shape", y_test.shape) # ([2, 4339610])

                # Get the predictions of our model.
                y_hat_test = model(x_test)
                # print("mss_y_hat_test", y_hat_test)
                # print("mss_y_hat_test.shape", y_hat_test.shape)
                # print("type(mss_y_hat_test)", type(y_hat_test))

                # print("y_hat_test.detach().numpy()", y_hat_test.detach().numpy().shape) # y_hat_test.detach().numpy() (2, 1025, 60)
                # predicted_chunks_features.append(y_hat_test.detach().numpy())
                if i == 0:
                    predicted_chunks_features = y_hat_test.detach().numpy()
                else:
                    predicted_chunks_features = np.append(predicted_chunks_features, y_hat_test.detach().numpy(),
                                                          axis=0)

        print("mss_predicted_chunks_features.shape", predicted_chunks_features.shape)

        # to_audio (mix_waveform: np.ndarray,
        #              predicted_vectors: np.ndarray)
        # :param mix_waveform: The waveform of the monaural mixture. Expected shape (n_samples,)
        # :type mix_waveform: numpy.ndarray
        # :param predicted_vectors: A numpy array of shape: (chunks, frequency_bins, time_frames)
        # :type predicted_vectors: numpy.ndarray
        # :return: predicted_waveform: The waveform of the predicted signal: (~n_samples,)
        # :rtype: numpy.ndarray

        predicted_waveform = to_audio(y_test.detach().numpy()[0], predicted_chunks_features)
        print("predicted_waveform.shape", predicted_waveform.shape)  # (4299776,)
        # y, sr = librosa.load("Dataset/Mixtures/testing/testing_1/mixture.wav")
        y, sr = librosa.load("Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav")  # 4339610

        sf.write('predicted.wav', predicted_waveform, samplerate=sr)
        print("MSE: %f" % np.mean((predicted_waveform - y_test.detach().numpy()[0][:len(predicted_waveform)]) ** 2.))  #
        # 0.021

        # Evaluation
        (sdr, isr, sir, sar, perm) = mir_eval.separation.bss_eval_images_framewise(
            reference_sources=y[0:4299776],
            estimated_sources=predicted_waveform,
            window=1*sr, hop=sr)
        # signal to interference ratio (SIR),
        # signal to artifacts ratio (SAR),
        # and signal to distortion ratio (SDR).
        # print("signal to interference ratio (SIR)", sir)
        # print("signal to artifacts ratio (SAR)", sar)
        # print("signal to distortion ratio (SDR)", sdr)

        # Explanation
        # I am having a noisy predicted.wav file instead of getting separated audio file. Current implementation of the
        # code does the testing on training data which is for the mixture in 053 - Actions - Devil's Words folder.
        # However, I cannot find the reason why I am getting noisy file.
        # Also, instead of data_handling.py  file which is mentioned in the assignment, I have created
        # dataset_class.py file. The only differencef is naming of the file. Contents are the same.


if __name__ == '__main__':
    main()

# EOF
