# Import necessary libraries
import os
from collections.abc import Iterable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import pandas as pd
from joblib import Memory
from joblib import delayed
from tqdm import tqdm
from utils.get_h5_data import get_h5_size
from utils.parallel_bar import ParallelExecutor
from config import Config
import scipy.signal
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Define a Dataset class for PSG (Polysomnography) data
class PSG_Dataset(Dataset):
    def __init__(self, config, mode):
        # Initialize dataset with configuration and mode (train, val, test)
        self.config = config  # Configuration object containing dataset and training parameters
        self.mode = mode  # Specifies the dataset mode: 'train', 'val', or 'test'
        # Cache path where processed data will be stored
        self.cachepath = os.path.join(self.config.cache_dir, mode + '_cache')
        # Directory where the training files are located
        self.filepath = os.path.join(self.config.train_dir)
        # List all files in the training directory
        self.filenames_all = [f for f in os.listdir(self.filepath) if os.path.isfile(os.path.join(self.filepath, f))]
        # Split data into training, validation, and testing sets
        self.data_split = self.train_val_test_split()
        # Files corresponding to the current mode (train, val, or test)
        self.filenames = self.data_split[self.mode]
        # Load dataset metadata from a CSV file
        self.dataframe = pd.read_csv(self.config.csv_dataset)
        # Normalize features based on the training set
        self.features_norm = self.get_features_norm()

        # Dataset attributes
        self.num_records = len(self.filenames)  # Number of records in the current mode
        self.label_name = config.label  # Name of the label column in the metadata
        self.epoch_size = config.epoch_size  # Size of each epoch (segment of data)
        self.n_channels = config.n_channels  # Number of channels in the data
        self.n_class = config.n_class  # Number of classes for classification
        self.lr_finder = False  # Flag for learning rate finder mode
        self.channel_drop = config.channel_drop  # Whether to randomly drop channels
        self.channel_drop_p = config.channel_drop_prob  # Probability of dropping a channel
        self.only_sleep_data = config.only_sleep  # Whether to use only sleep data
        self.only_eeg = config.only_eeg  # Whether to use only EEG channels

        # Load and process data
        self.cache_data = False  # Flag to determine whether to cache processed data
        self.n_jobs = 4  # Number of parallel jobs for data loading
        self.psgs = {}  # Dictionary to store PSG data information
        self.attrs = {}  # Dictionary to store attributes of each record
        self.features = {}  # Dictionary to store features of each record
        self.hyp = {}  # Dictionary to store hypnogram (sleep stages) data
        get_data = get_h5_size  # Function to get size of the H5 data
        # Cache data if enabled
        if self.cache_data:
            memory = Memory(self.cachepath, mmap_mode='r', verbose=0)
            get_data = memory.cache(get_h5_size)
        # Load data in parallel
        print(f'Number of recordings: {self.num_records}')
        data = ParallelExecutor(n_jobs=self.n_jobs, prefer='threads')(total=len(self.filenames))(delayed(get_data)(
            filename=os.path.join(self.filepath, record)) for record in self.filenames)
        for record, (data_size, attrs) in zip(self.filenames, data):
            features = self.get_features(record[:-5])
            if not features:
                continue
            self.psgs[record] = {'length': data_size,
                                 'reduced_length': int(data_size // self.epoch_size)}
            self.attrs[record] = attrs
            self.features[record] = features

        # Generate indexes for data retrieval
        if self.config.only_sleep:
            self.indexes = []
            for i, record in enumerate(self.psgs.keys()):
                ar_sig = self.load_h5(record)
                for j in np.arange(self.psgs[record]['reduced_length']):
                    start = j * self.epoch_size
                    stop = (j + 1) * self.epoch_size
                    if ar_sig[start:stop].sum() / 38400 <= 0.1:
                        self.indexes.append(((i, record), (start, stop)))
        else:
            self.indexes = [((i, record), (j * self.epoch_size, (j + 1) * self.epoch_size)) for i, record in
                            enumerate(self.psgs.keys())
                            for j in np.arange(self.psgs[record]['reduced_length'])]

        self.check_h5_quality()

    # Split the filenames into training, validation, and test sets
    def train_val_test_split(self):
        list_train, list_test = train_test_split(self.filenames_all, test_size=0.15, random_state=0)
        val_idx = int(0.2 * len(list_train))
        list_val = list_train[:val_idx]
        list_train = list_train[val_idx:]
        return {'train': list_train, 'val': list_val, 'test': list_test}

    # Normalize feature values based on the training data
    def get_features_norm(self):
        train_data = self.data_split['train']
        ids = [int(d[:-5]) for d in train_data]
        df_train = self.dataframe[self.dataframe['ID'].isin(ids)]
        features_norm = {}
        for feature in self.config.features:
            features_norm[feature] = [df_train[feature].min(), df_train[feature].max()]
        return features_norm

    # Check for consistency and integrity of H5 files
    def check_h5_quality(self):
        has_error = False
        for idx, record in enumerate(self.attrs):
            if idx == 0:
                base_attrs = self.attrs[record]
            else:
                for key in base_attrs:
                    if key not in self.attrs[record]:
                        print('Error: missing key. Record: ', record)
                        has_error = True
                    elif type(base_attrs[key]) != type(self.attrs[record][key]):
                        print('Error: wrong data type. Key: ', key, '. Record: ', record)
                        has_error = True
                    if isinstance(self.attrs[record][key], Iterable):
                        if any(self.attrs[record][key] != self.attrs[record][key]):
                            print('Error: nan data type. Key: ', key, '. Record: ', record)
                            has_error = True
                    else:
                        if self.attrs[record][key] != self.attrs[record][key]:
                            print('Error: nan data type. Key: ', key, '. Record: ', record)
                            has_error = True

        if has_error:
            print('Error.')
        else:
            print('Data quality ensured.')

    # Load H5 data based on filename and position
    def load_h5(self, filename, position=None):
        with h5py.File(os.path.join(self.filepath, filename), "r", rdcc_nbytes=100 * 1024 ** 2) as f:
            if not position:
                return np.array(f['PSG'][-1, :])
            # Extract data chunk based on the selection criteria (EEG channels, etc.)
            if self.only_eeg == 1:
                data = np.array(f['PSG'][0:6, position[0]:position[1]])
            elif self.only_eeg == 2:
                data = np.array(f['PSG'][[0, 1, 2, 3, 4, 5, -2, -1], position[0]:position[1]])
            elif self.only_eeg == 3:
                data = np.array(f['PSG'][:, position[0]:position[1]])

            if self.config.spectrogram:
                _, _, Sxx = scipy.signal.spectrogram(data[0, :], fs=128, nperseg=64, noverlap=16)
                data = Sxx[:11, :]
        return data

    # Normalize and retrieve features for a given filename
    def get_features(self, filename):
        features = []
        for feature in self.config.features:
            val = self.dataframe[self.dataframe["ID"] == int(filename)][feature]
            f = val.astype(float).item()
            f = (f-self.features_norm[feature][0])/(self.features_norm[feature][1]-self.features_norm[feature][0])
            if f > 1:
                f = 1
            elif f < 0:
                f = 0
            features.append(f)
        return features

    # Return the length of the dataset
    def __len__(self):
        if self.config.include_features == 2:
            return len(self.filenames)
        else:
            return len(self.indexes)

    # Optionally drop channels from the data
    def drop_channel(self, data):
        if self.channel_drop and self.mode == 'train':
            drop_idx = np.random.rand(data.shape[0]) < self.channel_drop_p
            data[drop_idx] = 0.0
            data = data / (1.0 - self.channel_drop_p)
        return data

    # Retrieve an item from the dataset by index
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Output dict configuration based on whether to include features
        if self.config.include_features == 2:
            record = self.filenames[idx]
            attrs = self.attrs[record]
            features = self.features[record]
            label = attrs[self.label_name]
            out = {'fid': record,
                   'features': features,
                   'label': label}
        else:
            record = self.indexes[idx][0][1]
            position = [self.indexes[idx][1][0], self.indexes[idx][1][1]]
            data = self.load_h5(record, position)
            if self.config.only_sleep:
                data = data[:-1]
            attrs = self.attrs[record]
            features = self.features[record]
            label = attrs[self.label_name]
            data = self.drop_channel(data)
            out = {'fid': record,
                   'position': position,
                   'data': torch.from_numpy(data.astype(np.float32)),
                   'features': features,
                   'label': label,
                   'all_attrs': attrs}
        return out

if __name__ == '__main__':
    # Example usage of the PSG_Dataset class
    config = Config()  # Configuration object
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration
    dataset = 'train'  # Dataset mode
    self = PSG_Dataset(config, dataset)  # Instantiate the PSG_Dataset
    dl = DataLoader(self, shuffle=True, batch_size=16, num_workers=0, pin_memory=True, drop_last=True)  # DataLoader
    bar_data = tqdm(dl, total=len(dl), desc=f'Loss: {np.inf:.04f}')  # Progress bar for DataLoader
    batch = next(iter(bar_data))  # Get a batch of data
