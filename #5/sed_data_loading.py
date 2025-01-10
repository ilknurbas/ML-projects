#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

from getting_and_init_the_data import get_data_loader, get_dataset


def split_data(data_path, batch_size):

    train_loader = get_data_loader(dataset=get_dataset('training', data_path, key_class='labels'), batch_size=batch_size, shuffle=True)

    validation_loader = get_data_loader(dataset=get_dataset('validation', data_path, key_class='labels'), batch_size=batch_size,
                                        shuffle=True)

    test_loader = get_data_loader(dataset=get_dataset('testing', data_path, key_class='labels'), batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


def main():
    data_path = Path('sed_dataset')
    batch_size = 8

    train_loader, validation_loader, test_loader = split_data(data_path, batch_size)

    train_files = train_loader.dataset.files
    print('The number of total training files are : ')
    print(len(train_files))

    validation_files = validation_loader.dataset.files
    print('The number of total valoidation files are : ')
    print(len(validation_files))

    test_files = test_loader.dataset.files
    print('The number of total testing files are : ')
    print(len(test_files))

    # print('train_files',(train_files[0]))
    # print('train_files', len(train_files[0]['labels'][0]))

if __name__ == '__main__':
    main()

# EOF
