#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from typing import Tuple, Optional

import librosa
from torch.utils.data import Dataset
import numpy
from torch.utils.data import DataLoader

__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):

    def __init__(self, split: bool, source_out_path: Optional[str] = "", mix_in_path: Optional[str] = "",
                 mix_in: Optional[str] = "", mix_raw_data: Optional[str] = "",
                 in_: Optional[list] = 'in_', out_: Optional[list] = 'out_'
                 ) -> None:

        super().__init__()
        self.data = []
        self.split = split
        self.in_ = in_
        self.out_ = out_
        self.mix_in = mix_in
        #  Mixtures (inputs) and Sources (target outputs) directories.
        if split:  # true --> train
            filenames_serialize_source = os.listdir(source_out_path)
            filenames_serialize_mix = os.listdir(mix_in_path)

            # print("len(filenames_serialize_source)", len(filenames_serialize_source))
            # print("len(filenames_serialize_mix)", len(filenames_serialize_mix))

            for i in range(len(filenames_serialize_source)):
                path_in = mix_in_path + "/" + filenames_serialize_mix[i]
                with open(path_in, 'rb') as f:
                    temp_in = pickle.load(f)
                path_out = source_out_path + "/" + filenames_serialize_source[i]
                with open(path_out, 'rb') as f:
                    temp_out = pickle.load(f)

                temp_dict = {'in_': temp_in, 'out_': temp_out}

                self.data.append(temp_dict)

        else:  # false --> test
            # raw audio
            y, sr = librosa.load(mix_raw_data)
            filenames_serialize_mix_in = os.listdir(mix_in)
            for i in range(len(filenames_serialize_mix_in)):
                path_in = mix_in + "/" + filenames_serialize_mix_in[i]
                with open(path_in, 'rb') as f:
                    temp_in = pickle.load(f)
                temp_dict = {'in_': temp_in, 'out_': y}  # , 'out_': y
                self.data.append(temp_dict)

    def __len__(self) -> int:
        return len(self.data)
        # if self.split:
        #     return len(self.data)
        # else:
        # return 0

    # This is a function that returns one training example
    def __getitem__(self, item: int) -> Tuple[numpy.ndarray, int]:
        if self.split:
            item_returned = self.data[item]
            return item_returned[self.in_], item_returned[self.out_]
        else:
            item_returned = self.data[item]
            return item_returned[self.in_], item_returned[self.out_]


def main():
    # training
    dataset_train = MyDataset(True, source_out_path="Dataset/Sources/training_features",
                              mix_in_path="Dataset/Mixtures/training_features")
    data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, drop_last=False)
    print("data_loader_train", data_loader_train.__len__())

    for i, batch in enumerate(data_loader_train):
        x, y = batch
        # print("batch", batch)
        # print("x", x)
        # print("y", y)
        print("batch", len(batch))  # 2
        print("x", len(x))  # 4
        print("y", len(y))  # 4
        print("x.shape", x.shape)  # ([4, 1025, 60])
        print("y.shape", y.shape)  # ([4, 1025, 60])

        break;

    # testing
    dataset_test = MyDataset(False, mix_in="Dataset/Mixtures/testing_features",
                             # mix_raw_data="Dataset/Mixtures/testing/testing_1/mixture.wav"
                             mix_raw_data="Dataset/Mixtures/training/053 - Actions - Devil's Words/mixture.wav")

    data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, drop_last=False)
    print("data_loader_test", data_loader_test.__len__())
    for i, batch in enumerate(data_loader_test):
        x, y = batch
        # print("batch", batch)
        # print("x", x)
        # print("y", y)
        print("batch", len(batch))  # 2
        print("x", len(x))  # 4
        print("y", len(y))  # 4
        print("x.shape", x.shape)  # ([4, 1025, 60])
        print("y.shape", y.shape)  #([4, 4339610])

        break;


if __name__ == '__main__':
    main()

# EOF
