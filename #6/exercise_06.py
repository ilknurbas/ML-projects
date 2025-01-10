#!/usr/bin/env python
# -*- coding: utf-8 -*-

import librosa
import numpy as np

__docformat__ = 'reStructuredText'
__all__ = ['to_audio']


def to_audio(mix_waveform: np.ndarray,
             predicted_vectors: np.ndarray) \
        -> np.ndarray:
    """
	:param mix_waveform: The waveform of the monaural mixture. Expected shape (n_samples,)
    :type mix_waveform: numpy.ndarray
	:param predicted_vectors: A numpy array of shape: (chunks, frequency_bins, time_frames)
    :type predicted_vectors: numpy.ndarray
	:return: predicted_waveform: The waveform of the predicted signal: (~n_samples,)
    :rtype: numpy.ndarray
	"""
    # Pre-defined (I)STFT parameters
    win_size = 2048
    hop_size = win_size // 2
    win_type = 'hamm'

    # STFT analysis of waveform
    c_x = librosa.stft(mix_waveform, n_fft=win_size, win_length=win_size, hop_length=hop_size, window=win_type)
    # Phase computation
    phs_x = np.angle(c_x)
    # Get the number of time-frames
    tf = phs_x.shape[1]

    # Number of chunks/sequences
    n_chunks, fb, seq_len = predicted_vectors.shape
    p_end = seq_len * n_chunks
    # Reshaping
    # rs_vectors = np.reshape(predicted_vectors, (fb, p_end))
    rs_vectors = np.reshape(np.moveaxis(predicted_vectors, 0, 1), (fb, p_end))
    # Reconstruction
    if p_end > tf:
        # Appending zeros to phase
        c_vectors = np.hstack((phs_x, np.zeros_like(phs_x[:, :p_end - seq_len])))
    else:
        c_vectors = rs_vectors * np.exp(1j * phs_x[:, :p_end])
    # ISTFT
    predicted_waveform = librosa.istft(c_vectors, win_length=win_size, hop_length=hop_size, window=win_type)

    return predicted_waveform


def main():
    # Make a test
    seq_length = 60
    sig_len = 1898192
    mix_sig = np.random.normal(0, 0.8, (sig_len,))
    print("mix_sig", mix_sig.shape)  # (1898192,)
    ##
    y, sr = librosa.load("Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav")
    mix_sig = y
    ##
    c_x = librosa.stft(mix_sig, n_fft=2048, win_length=2048, hop_length=1024, window='hamm')
    chop_factor = c_x.shape[1] % seq_length
    new_time_frames = c_x.shape[1] - chop_factor
    print("new_time_frames", new_time_frames // seq_length)

    # A sketch for the chunked sequences, that contain the magnitude estimates of the signal
    r_vectors = np.reshape(np.abs(c_x[:, :-chop_factor]), (new_time_frames // seq_length, c_x.shape[0], seq_length))
    # print("r_vectors.shape", r_vectors.shape) # (30, 1025, 60)
    rec_wav = to_audio(mix_sig, r_vectors)
    # ...
    # r_vectors predicted olması lazım aslında
    # Sen bu functionu kullanırken model outputunu birleştireceksin öyle functiona yollayacaksın
    # mix_sig mesela testing_1/mixture.wav
    # mix_sig librosa.datadan aldığın şey --> buna sadece functionda ihtiyacımız var
    # mix_sig chunked versionı sen modele sokuyorsun
    # and mix_sig librosa.datadan aldığını functionda kulllanıyorsun

    print("MSE: %f" % np.mean((rec_wav - mix_sig[:len(rec_wav)]) ** 2.))  # 0.000000
    # print("rec_wav",rec_wav)
    print("mix_sig", mix_sig.shape)  # mix_sig (4339610,)
    print("r_vectors", r_vectors.shape)  # (70, 1025, 60)
    print("rec_wav", rec_wav.shape)  # rec_wav (4299776,)

    y, sr = librosa.load("Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav")
    print("y.shape", y.data.shape)
    print("y.shape", y.shape)

    return None


if __name__ == '__main__':
    main()

# EOF
