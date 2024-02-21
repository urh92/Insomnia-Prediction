# Import necessary libraries and modules
import argparse  # For parsing command line arguments
import time  # For generating timestamps
from utils.str2bool import str2bool  # Custom utility for converting strings to boolean
import os  # For interacting with the operating system
import json  # For reading and writing JSON files
import numpy as np  # For numerical operations
import torch  # PyTorch library for deep learning
from torch.utils import data  # Utilities for data loading
from torch.utils.tensorboard import SummaryWriter  # For logging to TensorBoard
from config import Config  # Configuration class containing model and training settings
from trainer import Trainer  # Trainer class for handling training and validation
from models import BasicModel, M_PSG2FEAT, FeaturesNetwork  # Model classes
from psg_dataset import PSG_Dataset  # Dataset class for loading polysomnogram data
from utils.utils_loss import insomnia_loss  # Custom loss function for insomnia detection
from utils.util_json import NumpyEncoder  # Custom JSON encoder for handling numpy data types

# Seeds for replication, ensuring reproducibility of results
np.random.seed(0)
torch.manual_seed(0)

# Main function to setup and start the model training and evaluation
def main(args):
    # Determine the computing device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configuration setup
    config = Config()

    # Adjust configuration settings based on hyperparameters passed via command line arguments
    hyper_param_string = 'lr_{:.7f}_l2_{:.7f}_dof_{:.3f}_doc_{:.3f}_lf_{:.0f}_bs_{:.0f}_sl_{:.0f}_sd_{:.0f}_eeg_{:.0f}_'.format(
        *args.pre_hyperparam)
    
    # Update configuration settings with provided hyperparameters
    config.lr, config.l2, config.do_f, config.channel_drop_prob, _, config.batch_size, _ = args.pre_hyperparam[:7]
    config.epoch_size = int(args.pre_hyperparam[6] * 128 * 60)  # Epoch size calculated based on sequence length
    if args.pre_hyperparam[6] != 5:
        # Adjust batch size and padding length based on epoch size scale
        epoch_size_scale = args.pre_hyperparam[6] / 5
        config.batch_size = int(config.batch_size / epoch_size_scale)
        config.pad_length = int(config.pad_length / epoch_size_scale)
    
    # Adjust configurations based on whether to include only sleep data and EEG channels
    config.only_sleep = args.pre_hyperparam[7]
    config.only_eeg = args.pre_hyperparam[8]
    if config.only_eeg in [1, 2, 3]:
        config.n_channels = {1: 6, 2: 8, 3: 12}[config.only_eeg]
    if config.only_sleep and config.n_channels != 12:
        config.n_channels += 1
    if config.spectrogram == 1:
        config.n_channels = 10

    # Setup TensorBoard writer for logging
    writer_name = 'runs/nInsomnia_' + hyper_param_string + time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(writer_name)

    # Dataloader parameters setup
    train_params = {'batch_size': config.batch_size, 'num_workers': config.n_workers, 'pin_memory': True}

    # Initialize datasets and dataloaders for training, validation, and testing
    datasets, dataloaders = {}, {}
    for subset in ['train', 'val', 'test']:
        datasets[subset] = PSG_Dataset(config, subset)
        dataloaders[subset] = data.DataLoader(datasets[subset], shuffle=True if subset == 'train' else False,
                                              drop_last=True, **train_params)

    # Model selection based on configuration
    if config.include_features:
        model = FeaturesNetwork(config).to(device)
    else:
        model = M_PSG2FEAT(config).to(device)

    # Prepare for training
    config.save_dir = config.model_F_path
    labels_train = np.array([datasets['train'].attrs[key]['label'] for key in datasets['train'].attrs.keys()])
    _, counts = np.unique(labels_train, return_counts=True)
    pos_weight = torch.tensor([counts[0] / counts[1]]).to(device)

    # Initialize the loss function with positional weights for imbalance handling
    loss_fn = insomnia_loss(device, config.loss_func, pos_weight)
    # Initialize the trainer
    trainer = Trainer(model, loss_fn, config, writer, device=device, num_epochs=config.max_epochs, patience=config.patience, resume=args.resume)

    # Start training and validation
    trainer.train_and_validate(dataloaders['train'], dataloaders['val'])
    print('\nFinished training')

    # Evaluation on the test set
    metrics, predictions = trainer.evaluate_performance(dataloaders['test'], len(dataloaders['train']))
    print('\nFinished evaluation')
    # Save evaluation metrics and predictions
    metrics.to_csv(os.path.join(config.model_dir, 'metrics.csv'))
    with open(os.path.join(config.model_dir, 'predictions.json'), 'w') as fp:
        json.dump(predictions, fp, sort_keys=True, indent=4, cls=NumpyEncoder)

    # Save features and predictions if specified
    if args.save_feat:
        if config.only_sleep == 1:
            config.only_sleep = 0  # Adjust configuration for feature saving
            datasets, dataloaders = {}, {}
            for subset in ['train', 'val', 'test']:
                datasets[subset] = PSG_Dataset(config, subset)
                dataloaders[subset] = data.DataLoader(datasets[subset], shuffle=True if subset == 'train' else False, drop_last=True, **train_params)

        trainer.save_features(dataloaders)  # Save the features

        writer.close()  # Close the TensorBoard writer
        return

# Argument parser setup for handling command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Insomnia Detection from Polysomnograms.')
    # Define arguments for pre-training, hyperparameters, model resuming, testing, saving features, training, Bayesian optimization, and resuming training
    parser.add_argument('--pre_train', type=str2bool, default=True, nargs='?', const=True, help='To pretrain model.')
    parser.add_argument('--pre_hyperparam', nargs=9, type=float, default=[1e-4, 1e-5, 0.5, 0.1, 1, 16, 5, 0, 2], help='Pretraining hyperparameters [learning rate, l2, dropout features, dropout channels, loss function, batch size, sequence length, only sleep data, only eeg].')
    parser.add_argument('--resume', type=str2bool, default=False, nargs='?', const=True, help='To resume previously pretrained model')
    parser.add_argument('--test_pre', type=str2bool, default=True, nargs='?', const=True, help='To train previously pretrained model')
    parser.add_argument('--save_feat', type=str2bool, default=False, nargs='?', const=True, help='Save/overwrite model F features')
    parser.add_argument('--train', type=str2bool, default=True, nargs='?', const=True, help='To train model.')
    parser.add_argument('--bo', type=str2bool, default=False, nargs='?', const=True, help='To perform Bayesian hyperparameter optimization.')
    parser.add_argument('--train_resume', type=str2bool, default=False, nargs='?', const=True, help='To resume previously trained model')
    parser.add_argument('--test', type=str2bool, default=True, nargs='?', const=True, help='To test previously trained model')

    args = parser.parse_args()  # Parse arguments
    print(args)
    main(args)  # Call the main function with parsed arguments
