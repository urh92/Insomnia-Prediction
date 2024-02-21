import os
import numpy as np

#Configuration class that contain attributes to set paths and network options.
class Config(object):
    def __init__(self):
        # Get profile
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        with open(os.path.join(__location__, 'profile.txt'), 'r') as f:
            profile = f.readline()

        # Set local data directory
        if profile == 'local':
            self.overall_dir = 'C:/Users/UmaerHANIF/Documents'
            self.data_dir = os.path.join(self.overall_dir, 'H5py Files')
            self.excel_dir = os.path.join(self.overall_dir, 'Excel')
            self.csv_dataset = os.path.join(self.excel_dir, 'Dataset_All.csv')
        elif profile == 'sherlock':
            self.overall_dir = '/scratch/users/umaer/Insomnia'
            self.data_dir = os.path.join(self.overall_dir, 'H5py Files')
            self.excel_dir = os.path.join(self.overall_dir, 'Excel')
            self.csv_dataset = os.path.join(self.excel_dir, 'Dataset_All.csv')

        # Datapaths
        self.model_dir = os.path.join(self.overall_dir, 'Model')
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'val')
        self.interp_dir = os.path.join(self.data_dir, 'interpretation')
        self.am_dir = os.path.join(self.data_dir, 'am')
        self.test_dir = os.path.join(self.data_dir, 'test')
        self.cache_dir = 'C:/Users/UmaerHANIF/Documents/Cache'
        self.train_cache_dir = os.path.join(self.cache_dir, 'train_cache')
        self.val_cache_dir = os.path.join(self.cache_dir, 'val_cache')
        self.test_cache_dir = os.path.join(self.cache_dir, 'test_cache')
        self.train_F_dir = os.path.join(self.data_dir, 'train_F')
        self.val_F_dir = os.path.join(self.data_dir, 'val_F')
        self.test_F_dir = os.path.join(self.data_dir, 'test_F')
        self.train_dir = os.path.join(self.data_dir)
        self.F_train_dir = os.path.join(self.data_dir, 'all_F')

        # Checkpoint
        self.save_dir = self.model_dir
        self.model_F_path = os.path.join(self.model_dir)

        # training
        # label-config
        self.label = 'label'
        self.label_size = [1]  # [1, 1, 2]
        self.n_class = 2
        self.only_sleep = 0
        # network-config
        self.n_channels = 8
        self.model_num = 1
        # train-config
        self.max_epochs = 10
        self.patience = 3
        self.batch_size = 32
        self.lr = 1e-3
        self.n_workers = 0
        self.do_f = 0.4
        self.channel_drop = True
        self.channel_drop_prob = 0.1
        self.loss_func = 'cross-entropy'
        self.only_eeg = 2
        self.l2 = 1e-5
        self.pad_length = 120
        # network-config
        self.net_size_scale = 4
        self.lstm_n = 1
        self.epoch_size = 5 * 60 * 128
        self.return_att_weights = False
        self.features = ["DisplayGender", "TST", "WASO", "SE", "SME", "AHI", "AI", 'Arousal Max', '3-5', '5-15',
                         '15-30', '30-60', '60-90', '90-120', '120-150', '150-180', '180-300', '300-600', '600-1800',
                         '1800-3600', '3600+']
        self.include_features = 0
        self.spectrogram = 0