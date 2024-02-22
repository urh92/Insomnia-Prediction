# Import necessary libraries and modules
import copy  
import os  
import numpy as np  
import pandas as pd 
import torch 
from tqdm import tqdm

# Define a BaseTrainer class to encapsulate training logic
class BaseTrainer(object):
    def __init__(self, network, loss_fn, config, device='cpu', num_epochs=100, patience=None, resume=None, scheduler=None):
        super().__init__()
        # Configuration and model parameters
        self.config = config  # Configuration object containing model settings
        self.network = network  # The neural network to train
        self.loss_fn = loss_fn  # Loss function used for training
        self.device = device  # Device to run the training on (CPU or GPU)
        self.num_epochs = num_epochs  # Total number of epochs to train for
        self.patience = patience if patience else self.num_epochs  # Patience for early stopping
        self.resume = resume  # Path to a checkpoint to resume training from
        self.save_dir = config.save_dir  # Directory where to save checkpoints
        self.scheduler = scheduler  # Learning rate scheduler
        self.l2 = config.l2  # L2 regularization factor
        self.lr = config.lr  # Learning rate

        # If resuming from checkpoint, load network and optimizer state
        if self.resume:
            self.checkpoint = torch.load(os.path.join(self.save_dir, 'latest_checkpoint.tar'))
            self.network.load_state_dict(self.checkpoint['network_state_dict'])

        # Move network to the specified device and initialize the optimizer
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(
            [{'params': [p for name, p in self.network.named_parameters() if 'weight' in name], 'weight_decay': self.l2},
             {'params': [p for name, p in self.network.named_parameters() if 'weight' not in name], 'weight_decay': 0}], lr=self.lr)

        # Load optimizer state if resuming
        if self.resume:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        # Lists to keep track of losses
        self.train_losses = []
        self.eval_losses = []

    def on_begin_epoch(self):
        # Display the current epoch number
        print(f'\nEpoch nr. {self.current_epoch + 1} / {self.num_epochs}')

    def on_end_epoch(self):
        # Update best model and reset patience if there is improvement
        if self.eval_losses[-1] < self.best_loss:
            self.best_loss = self.eval_losses[-1]
            self.best_network = copy.deepcopy(self.network)
            self.last_update = self.current_epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Save a checkpoint at the end of each epoch
        if self.save_dir:
            self.save_checkpoint(os.path.join(self.save_dir, 'latest_checkpoint.tar'))

    def on_begin_training(self):
        # Prepare the network for training
        self.network.train()

    def on_begin_validation(self):
        # Prepare the network for evaluation
        self.network.eval()

    def on_begin(self):
        # Initialization before training starts
        self.best_loss = np.inf
        self.best_network = None
        self.patience_counter = 0
        self.current_epoch = 0
        self.last_update = None
        self.start_epoch = 0

    def on_end(self):
        # Finalize training, save history, and perform the last epoch's end actions
        self.on_end_epoch()
        pd.DataFrame({'train': self.train_losses, 'eval': self.eval_losses}).to_csv(os.path.join(self.save_dir, 'history.csv'))

    def save_checkpoint(self, checkpoint_path):
        # Save the current state of training to a checkpoint
        checkpoint = {'best_loss': self.best_loss,
                      'best_network': self.best_network,
                      'start_epoch': self.current_epoch,
                      'last_update': self.last_update,
                      'network_state_dict': self.network.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'patience_counter': self.patience_counter,
                      'train_losses': self.train_losses,
                      'eval_losses': self.eval_losses}
        torch.save(checkpoint, checkpoint_path)

    def restore_checkpoint(self):
        # Restore training state from a checkpoint
        self.best_loss = self.checkpoint['best_loss']
        self.best_network = self.checkpoint['best_network']
        self.patience_counter = self.checkpoint['patience_counter']
        self.start_epoch = self.checkpoint['start_epoch'] + 1
        self.last_update = self.checkpoint['last_update']
        self.train_losses = self.checkpoint['train_losses']
        self.eval_losses = self.checkpoint['eval_losses']

    # Placeholder methods to be implemented by subclasses
    def train_and_validate(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def validate_step(self):
        raise NotImplementedError
