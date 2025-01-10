import os
import pickle

import librosa
import numpy as np


def create_sub_directories():
    mixture_folder_train = "Dataset/Mixtures/training_features"
    mixture_folder_test = "Dataset/Mixtures/testing_features"
    sources_folder_train = "Dataset/Sources/training_features"
    sources_folder_test = "Dataset/Sources/testing_features"
    folders = [mixture_folder_train, mixture_folder_test, sources_folder_train, sources_folder_test]
    for folder in folders:
        if not os.path.isdir(folder):
            os.makedirs(folder)


def extract_stft(sig_path):  # sig
    y, sr = librosa.load(sig_path)

    seq_length = 60
    c_x = librosa.stft(y, n_fft=2048, win_length=2048, hop_length=1024, window='hamm')
    # print("c_x.shape[0]", c_x.shape[0])  # frequency bin
    # print("c_x.shape[1]", c_x.shape[1])  # frame number
    chop_factor = c_x.shape[1] % seq_length
    new_time_frames = c_x.shape[1] - chop_factor  # new number of frames
    r_vectors = np.reshape(np.abs(c_x[:, :-chop_factor]), (new_time_frames // seq_length, c_x.shape[0], seq_length))

    # print("r_vectors", r_vectors.shape)  # r_vectors (80, 1025, 60)
    # print("r_vectors", r_vectors)
    # print("r_vectors", len(r_vectors))  # 80
    # print("r_vectors", len(r_vectors[0]))  # 1025
    # print("r_vectors", len(r_vectors[0][0]))  # 60
    return r_vectors


def serialize_stft(r_vectors, f_output):
    output = open(f_output, 'wb')
    pickle.dump(r_vectors, output, -1)
    output.close()


def save_serialized_stft():

    # mixtures - training
    mix_train_path_1 = "Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav"
    mix_train_path_2 = "Dataset/Mixtures/training/054 - Actions - South Of The Water/mixture.wav"
    mix_train = [mix_train_path_1, mix_train_path_2]
    song_name = ["053-Actions-Devil's-Words", "054-Actions-South-Of-The-Water"]
    for m_t in range(len(mix_train)):
        split_stft = extract_stft(mix_train[m_t])
        for split in range(split_stft.shape[0]):
            out_path = "Dataset/Mixtures/training_features/" + "mix_" + song_name[m_t] + "_seq_" + str(
                split + 1) + ".npy"
            serialize_stft(split_stft[split], out_path)

    # sources - training
    sources_train_path_1 = "Dataset/Sources/training/053 - Actions - Devil's Words/vocals.wav"
    sources_train_path_2 = "Dataset/Sources/training/054 - Actions - South Of The Water/vocals.wav"
    source_train = [sources_train_path_1, sources_train_path_2]
    song_name = ["053-Actions-Devil's-Words", "054-Actions-South-Of-The-Water"]
    for s_t in range(len(source_train)):
        split_stft = extract_stft(source_train[s_t])
        for split in range(split_stft.shape[0]):
            out_path = "Dataset/Sources/training_features/source_" + song_name[s_t] + "_seq_" + str(split + 1) + ".npy"
            serialize_stft(split_stft[split], out_path)

    # mixtures - testing
    # mix_test_path_1 = "Dataset/Mixtures/testing/testing_1/mixture.wav"
    # mix_test_path_2 = "Dataset/Mixtures/testing/testing_2/mixture.wav"
    mix_test_path_1 = "Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav"
    # mix_train_path_1 = "Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav"
    mix_test_path_1 = mix_train_path_1
    split_stft = extract_stft(mix_test_path_1)
    for split in range(split_stft.shape[0]):
        out_path = "Dataset/Mixtures/testing_features/mix_test1_seq_" + str(split + 1) + ".npy"
        # out_path = "Dataset/Mixtures/testing_features/mix_test2_seq_" + str(split + 1) + ".npy"
        serialize_stft(split_stft[split], out_path)

    # sources - testing
    source_test_path_1 = "Dataset/Sources/testing/testing_1/vocals.wav"
    # source_test_path_2 = "Dataset/Sources/testing/testing_2/vocals.wav"
    source_test_path_1 = sources_train_path_1
    split_stft = extract_stft(source_test_path_1)
    for split in range(split_stft.shape[0]):
        out_path = "Dataset/Sources/testing_features/vocals_test1_seq_" + str(split + 1) + ".npy"
        # out_path = "Dataset/Sources/testing_features/vocals_test2_seq_" + str(split + 1) + ".npy"
        serialize_stft(split_stft[split], out_path)

def main():
    # create directory
    create_sub_directories()
    # save the serialized versions to respective folders
    save_serialized_stft()


if __name__ == '__main__':
    main()

# EOF
