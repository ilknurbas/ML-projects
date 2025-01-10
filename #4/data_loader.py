from dataset_class import MyDataset
from torch.utils.data import DataLoader
from pathlib import Path


def load_data(data_path, split, batch_size, shuffle, drop_last, num_workers):

    # Create an object of class MyDataset and give it the data folder full path to read data from.
    dataset = MyDataset(split, data_path)
    # Create a DataLoader object and pass it as input the "dataset" object and other needed input parameters
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             drop_last=drop_last, num_workers=num_workers)

    return data_loader


def main():
    batch_size = 8
    # data_path = Path('../music_speech_dataset')
    data_path = Path('music_speech_dataset')

    # loading the training data
    print('Loading the training data')
    split = "training"
    train_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)
    train_files = train_loader.dataset.files
    print('The number of total training files are : ')
    print(len(train_files))

    # load the validation data
    print('Loading the validation data')
    split = "validation"
    validation_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)
    validation_files = validation_loader.dataset.files
    print('The number of total valoidation files are : ')
    print(len(validation_files))

    # load the testing data
    print('Loading the testing data')
    split = "testing"
    test_loader = load_data(data_path, split, batch_size, shuffle=False, drop_last=True, num_workers=1)
    test_files = test_loader.dataset.files
    print('The number of total testing files are : ')
    print(len(test_files))
    print((test_files[0]['features']).shape)


if __name__ == '__main__':
    main()

# EOF
