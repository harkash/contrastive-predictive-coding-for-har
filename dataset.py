import numpy as np
import os
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

from sliding_window import sliding_window


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))

    # Just making it a vector if it was a 2D matrix
    data_y = np.reshape(data_y, (len(data_y),))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)). \
        astype(np.uint8)


# Defining the data loader for the implementation
class HARDataset(Dataset):
    def __init__(self, args, phase):
        self.filename = os.path.join(args.root_dir, args.data_file)
        print(self.filename)

        # If the prepared dataset doesn't exist, give a message and exit
        if not os.path.isfile(self.filename):
            print('The data is not available. '
                  'Ensure that the data is present in the directory.')
            exit(0)

        # Loading the data from the .mat file
        self.data_raw = self.load_dataset(self.filename)
        assert args.input_size == self.data_raw[phase]['data'].shape[1]

        # Obtaining the segmented data
        self.data, self.labels = \
            opp_sliding_window(self.data_raw[phase]['data'],
                               self.data_raw[phase]['labels'],
                               args.window, args.overlap)

    def load_dataset(self, filename):
        data = loadmat(filename)
        data_raw = {'train': {'data': data['X_train'],
                              'labels': np.transpose(data['y_train'])},
                    'val': {'data': data['X_valid'],
                            'labels': np.transpose(data['y_valid'])},
                    'test': {'data': data['X_test'],
                             'labels': np.transpose(data['y_test'])}}

        for set in ['train', 'val', 'test']:
            data_raw[set]['data'] = data_raw[set]['data'].astype(np.float32)
            data_raw[set]['labels'] = data_raw[set]['labels'].astype(np.uint8)

        return data_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index, :, :]

        data = torch.from_numpy(data).double()

        label = torch.from_numpy(np.asarray(self.labels[index])).double()
        return data, label


def load_dataset(args, classifier=False):
    datasets = {x: HARDataset(args=args, phase=x) for x in
                ['train', 'val', 'test']}

    def get_batch_size():
        if classifier:
            batch_size = args.classifier_batch_size
        else:
            batch_size = args.batch_size

        return batch_size

    data_loaders = {x: DataLoader(datasets[x],
                                  batch_size=get_batch_size(),
                                  shuffle=True if x == 'train' else False,
                                  num_workers=2, pin_memory=True)
                    for x in ['train', 'val', 'test']}

    # Printing the batch sizes
    for phase in ['train', 'val', 'test']:
        print('The batch size for {} phase is: {}'
              .format(phase, data_loaders[phase].batch_size))

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}
    print(dataset_sizes)

    return data_loaders, dataset_sizes


