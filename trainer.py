# Import necessary libraries
import os
from config import Config
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
from base.base_trainer import BaseTrainer
from utils.util_am import am_model
from torch.utils.data import Dataset, DataLoader
from psg_dataset import PSG_Dataset
from captum.attr import (GradientShap, DeepLift, DeepLiftShap, InputXGradient, IntegratedGradients, NoiseTunnel)
from utils.utils_loss import insomnia_loss
from models import BasicModel

# Define the Trainer class, inheriting from BaseTrainer, to handle the training process
class Trainer(BaseTrainer):
    def __init__(self, network, loss_fn, config, writer=None, device='cpu', num_epochs=100, patience=None, resume=None, scheduler=None):
        # Initialize the trainer with network, loss function, configurations, and training options
        super().__init__(network, loss_fn, config, device=device, num_epochs=num_epochs, patience=patience, resume=resume, scheduler=scheduler)
        self.writer = writer  # TensorBoard writer for logging
        self.iter = 0  # Iteration counter
        self.len_epoch = 0  # To keep track of the number of batches per epoch
        self.include_features = config.include_features  # Whether to include additional features in the model input

    def train_and_validate(self, train_dataloader, eval_dataloader):
        # Method to handle the training and validation process
        self.on_begin()  # Perform setup operations at the start of training

        self.len_epoch = len(train_dataloader)  # Determine the number of batches in the training dataset

        if self.resume:
            # If resuming from a checkpoint, restore the model and optimizer states
            self.restore_checkpoint()
            self.iter = self.current_epoch * self.len_epoch // self.config.batch_size

        # Main training loop
        for self.current_epoch in range(self.start_epoch, self.num_epochs):
            self.on_begin_epoch()  # Setup operations for each epoch

            # Switch model to training mode
            self.network.train()
            batch_losses = []  # List to store losses for each batch
            bar_train = tqdm(train_dataloader, total=self.len_epoch, desc=f'Train Loss: {np.inf:.04f}')
            for batch in bar_train:
                # Process each batch from the training dataloader
                loss_out = self.train_step(batch)  # Perform a training step
                loss = loss_out.item()  # Extract scalar loss value
                batch_losses.append(loss)  # Append the loss to the list for averaging

                # Update the progress bar with the current average loss
                bar_train.set_description(f'Train Loss: {np.mean(batch_losses):.04f}')

                self.iter += 1  # Increment the iteration counter
                self.log_metrics_to_tensorboard(loss_out, 'Training')  # Log training metrics to TensorBoard

            self.train_losses.append(np.mean(batch_losses))  # Record the average training loss for the epoch

            # Switch model to evaluation mode for validation
            self.network.eval()
            batch_losses = []  # Reset list to store losses for each validation batch
            batch_loss_outs = []  # List to store loss tensors for logging
            bar_eval = tqdm(eval_dataloader, total=len(eval_dataloader), desc=f'Val Loss: {np.inf:.04f}')
            with torch.no_grad():  # Disable gradient computation during validation
                for batch in bar_eval:
                    # Process each batch from the validation dataloader
                    loss_out = self.eval_step(batch)  # Perform a validation step
                    loss = loss_out.item()  # Extract scalar loss value
                    batch_losses.append(loss)  # Append the loss to the list for averaging

                    # Update the progress bar with the current average validation loss
                    bar_eval.set_description(f'Val Loss: {np.mean(batch_losses):.04f}')

                self.eval_losses.append(np.mean(batch_losses))  # Record the average validation loss for the epoch

            self.log_metrics_to_tensorboard(batch_loss_outs, 'Validation')  # Log validation metrics to TensorBoard

            self.on_end_epoch()  # Cleanup operations at the end of each epoch

            if self.patience_counter > self.patience:
                # If early stopping criterion is met, exit the training loop
                print('\nEarly stopping criterion reached, stopping training!')
                break

        self.on_end()  # Cleanup operations at the end of training
        
    def train_step(self, batch):
        if self.include_features == 2:
            features, y = torch.stack(batch["features"]).transpose(1, 0).to(self.device), batch['label'].to(self.device)
        else:
            x, y = batch['data'].to(self.device), batch['label'].to(self.device)#.type(torch.LongTensor).to(self.device)
            features = torch.stack(batch["features"]).transpose(1, 0).to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

        # Run forward pass
        if self.include_features == 0:
            y_p = self.network(x).squeeze()
        elif self.include_features == 1:
            y_p = self.network(x, features).squeeze()
        else:
            y_p = self.network(features).squeeze()

        # Calculate training loss
        loss_out = self.loss_fn.loss(y_p, y)
        loss = loss_out.item()

        # Run optimization
        loss_out.backward()
        self.optimizer.step()
        return loss_out

    def eval_step(self, batch):
        if self.include_features == 2:
            features, y = torch.stack(batch["features"]).transpose(1, 0).to(self.device), batch['label'].to(self.device)
        else:
            x, y = batch['data'].to(self.device), batch['label'].to(self.device)#.type(torch.LongTensor).to(self.device)
            features = torch.stack(batch["features"]).transpose(1, 0).to(self.device)

        # Run forward pass
        if self.include_features == 0:
            y_p = self.network(x).squeeze()
        elif self.include_features == 1:
            y_p = self.network(x, features).squeeze()
        else:
            y_p = self.network(features).squeeze()

        # Calculate training loss
        loss_out = self.loss_fn.loss(y_p, y)
        return loss_out

    def check_nan(self, y, y_p, loss):
        # Check y
        check_y = any([x.item() != x.item() for x in y])
        # Check y_p
        check_y_p = any([x.item() != x.item() for x in y_p])
        # Check loss
        check_loss = loss.item() != loss.item()
        return any([check_y, check_y_p, check_loss])

    def predict_step(self, batch):
        if self.include_features == 2:
            record = batch['fid']
            features, y = torch.stack(batch["features"]).transpose(1, 0).to(self.device), batch['label'].to(self.device)
        else:
            record, position = batch['fid'], batch['position']
            x, y = batch['data'].to(self.device), batch['label'].to(self.device)#.type(torch.LongTensor).to(self.device)
            features = torch.stack(batch["features"]).transpose(1, 0).to(self.device)

        # Run forward pass
        if self.include_features == 0:
            y_p = self.best_network(x).squeeze()
        elif self.include_features == 1:
            y_p = self.best_network(x, features).squeeze()
        else:
            y_p = self.best_network(features).squeeze()
        out = {'pred': y_p}

        # Add record and label info
        out['fids'] = record
        out['label'] = y
        if self.include_features != 2:
            out['position'] = position
        return out

    def evaluate_performance(self, test_dataloader, len_epoch=0):
        self.len_epoch = len_epoch

        if self.resume:
            self.restore_checkpoint()
            self.iter = self.last_update * self.len_epoch // self.config.batch_size

        # initialize records
        records = [record for record in test_dataloader.dataset.filenames]
        predictions = {r: {'insomnia_p': [], 'pos': [], 'label': []} for r in records}
        metrics = {'record': records,
                   'insomnia': [],
                   'insomnia_p': []}

        print(f'\nEvaluating model')
        self.best_network.eval()
        batch_losses = []
        batch_loss_outs = []
        bar_test = tqdm(test_dataloader, total=len(test_dataloader))
        with torch.no_grad():
            for batch in bar_test:
                fids = batch['fid']
                y = batch['label'].to(self.device)#.type(torch.LongTensor).to(self.device)
                features = torch.stack(batch["features"]).transpose(1, 0).to(self.device)
                if self.include_features != 2:
                    pos = batch['position']
                    x = batch['data'].to(self.device)

                # Run forward pass and get predictions
                if self.include_features == 0:
                    y_p = self.best_network(x).squeeze()
                elif self.include_features == 1:
                    y_p = self.best_network(x, features).squeeze()
                else:
                    y_p = self.best_network(features).squeeze()

                # Compute loss
                loss_out = self.loss_fn.loss(y_p, y)
                loss = loss_out.item()
                batch_losses.append(loss)
                batch_loss_outs.append(loss_out)

                # Assign to subjects
                if self.include_features == 2:
                    for record, pred, labels in zip(fids, y_p, y):
                        predictions[record]['insomnia_p'].append(torch.sigmoid(pred).item())
                        predictions[record]['label'].append(labels.item())
                else:
                    for record, r_pos, pred, labels in zip(fids, pos[0], y_p, y):
                        # Predictions
                        predictions[record]['pos'].append(r_pos.numpy().tolist())
                        predictions[record]['insomnia_p'].append(torch.sigmoid(pred).item())
                        predictions[record]['label'].append(labels.item())

        # Log to tensorboard
        self.log_metrics_to_tensorboard(batch_loss_outs, 'Test')

        # Calculate metrics
        for record in metrics['record']:

            # Label
            y = predictions[record]['label']

            # If no data return NaN
            if len(y) == 0:
                metrics['insomnia'].append(np.NaN)
                metrics['insomnia_p'].append(np.NaN)
            else:
                # Average prediction
                y_p = [[np.mean(predictions[record]['insomnia_p'])]]

                # Log labels
                metrics['insomnia'].append(y[0])

                # Log averaged predictions
                metrics['insomnia_p'].append(y_p[0][0])
                # Log loss and accuracy

        return pd.DataFrame.from_dict(metrics).set_index('record'), predictions

    def save_features(self, dataloaders):
        if self.resume:
            self.restore_checkpoint()

        print(f'\nEvaluating features')
        self.best_network.eval()

        # Iterate dataloaders
        for k, dl in dataloaders.items():

            print(f'\nEvaluating subset: ', k)
            if k == 'train':
                dl.dataset.mode = 'save_feat'
                dl = data.DataLoader(dl.dataset, shuffle=False, batch_size=dl.batch_size, num_workers=dl.num_workers,
                                     pin_memory=dl.pin_memory)
            # initialize records
            records = [record for record in dl.dataset.filenames]
            feature_dict = {
                r: {'feat': [], 'label': [], 'label_cond': [], 'attrs': [], 'insomnia_p': []}
                for r in records}

            # Compute features for each batch
            bar_feat = tqdm(dl, total=len(dl))
            with torch.no_grad():
                for batch in bar_feat:
                    out = self.predict_step(batch)
                    attrs = batch['all_attrs'].copy()

                    # Collect features
                    for i in range(len(out['fids'])):

                        label = out['label'][i].cpu().numpy()
                        for key_a, v in batch['all_attrs'].items():
                            attrs[key_a] = v[i].cpu().numpy()
                        insomnia_p = out['pred'][i].cpu().numpy()

                        if feature_dict[out['fids'][i]]['label'] == []:
                            feature_dict[out['fids'][i]]['label'] = label
                            feature_dict[out['fids'][i]]['attrs'] = attrs

                        feature_dict[out['fids'][i]]['insomnia_p'].append(insomnia_p)

            # Save feature dict as h5
            for record, v in feature_dict.items():
                output_filename = os.path.join(self.config.F_train_dir, record)
                with h5py.File(output_filename, "w") as f:
                    # Add datasets
                    f.create_dataset("insomnia_p", data=np.stack(feature_dict[record]['insomnia_p']), dtype='f4')
                    # Attributes
                    for key_a, v in feature_dict[record]['attrs'].items():
                        f.attrs[key_a] = v

    def log_metrics_to_tensorboard(self, loss, name):
        if self.writer is not None:
            if isinstance(loss, list):
                self.writer.add_scalar(name + '/loss',
                                       np.mean([bl.item() for bl in loss]),
                                       self.iter)
            elif isinstance(loss, dict):
                self.writer.add_scalar(name + '/loss',
                                       np.array(loss['loss'].item()),
                                       self.iter)
            else:
                self.writer.add_scalar(name + '/loss',
                                       np.array(loss.item()),
                                       self.iter)
        return

    def activation_maximization(self, save_path=None, n_iter=1e4, in_size=[1, 12, 128 * 5 * 60], lr=1e-5, l2=1e-4):
        if self.resume:
            self.restore_checkpoint()

        self.best_network.eval()
        self.best_network.return_only_pred = True

        self.am_model = am_model(self.best_network, in_size).to(self.device)

        self.am_optimizer = torch.optim.Adam(
            [{'params': [p for name, p in self.am_model.named_parameters() if 'am_data' in name], 'weight_decay': l2},
             {'params': [p for name, p in self.am_model.named_parameters() if 'am_data' not in name],
              'weight_decay': 0}],
            lr=lr)

        self.am_model.train()
        for i in range(int(n_iter)):
            self.am_optimizer.zero_grad()
            output = self.am_model()
            loss = - output.squeeze()
            loss.backward()
            self.am_optimizer.step()
            if i % (n_iter // 100) == 0:
                print('loss: ', loss.item())

        am_data = self.am_model.am_data.detach().cpu().numpy()
        output_filename = os.path.join(save_path, 'am_5.hdf5')
        with h5py.File(output_filename, "w") as f:
            # Save PSG
            f.create_dataset("am_data", data=am_data, dtype='f4')
        return

    def interpret_model(self, test_dataloader, save_path=None, atr_method='int_grad'):
        if self.resume:
            self.restore_checkpoint()

        print(f'\nComputing model interpretation')
        # Model train to enable gradient computation
        self.best_network.eval()
        self.best_network.LSTM.training = True
        self.best_network.return_only_pred = True

        if atr_method == 'int_grad':
            i_model = IntegratedGradients(self.best_network)
        elif atr_method == 'grad_shap':
            i_model = GradientShap(self.best_network)
        elif atr_method == 'deep_lift':
            i_model = DeepLift(self.best_network)
        elif atr_method == 'deep_lift_shap':
            i_model = DeepLiftShap(self.best_network)
        elif atr_method == 'int_smooth_grad':
            i_model = IntegratedGradients(self.best_network)
            i_model = NoiseTunnel(i_model)
        elif atr_method == 'inputXgradient':
            i_model = InputXGradient(self.best_network)
        # elif atr_method == 'occlusion':

        bar_test = tqdm(test_dataloader, total=len(test_dataloader))
        current_subj = ''
        for batch in bar_test:

            fids = batch['fid']
            pos = batch['position']
            x = batch['data'].to(self.device)

            # Run interpretation
            if atr_method == 'int_grad':
                baseline = torch.zeros_like(x)
                attributions, delta = i_model.attribute(x, baseline, target=0, n_steps=10,
                                                        return_convergence_delta=True)
            elif atr_method == 'grad_shap':
                baseline_dist = torch.randn(x.size(0) * 5, x.size(1), x.size(2)).to(self.device) * 0.001
                attributions, delta = i_model.attribute(x, stdevs=0.09, n_samples=4, baselines=baseline_dist, target=0,
                                                        return_convergence_delta=True)
                delta = torch.mean(delta.reshape(x.shape[0], -1), dim=1)
            elif atr_method == 'deep_lift':
                baseline = torch.zeros_like(x)
                attributions, delta = i_model.attribute(x, baseline, target=0, return_convergence_delta=True)
            elif atr_method == 'deep_lift_shap':
                baseline_dist = torch.randn(x.size(0) * 5, x.size(1), x.size(2)).to(self.device) * 0.001
                attributions, delta = i_model.attribute(x, baseline_dist, target=0, return_convergence_delta=True)
                delta = torch.mean(delta.reshape(x.shape[0], -1), dim=1)
            elif atr_method == 'int_smooth_grad':
                baseline = torch.zeros_like(x)
                attributions, delta = i_model.attribute(x, baselines=baseline, nt_type='smoothgrad', stdevs=0.02,
                                                        target=0, n_samples=5, n_steps=5, return_convergence_delta=True)
            elif atr_method == 'inputXgradient':
                attributions = i_model.attribute(x, target=0)
                delta = torch.zeros_like(x)
            elif atr_method == 'occlusion':
                attributions, delta = self.occlusion_attribution(x)

            # Assign to subjects
            for record, b_pos, seq_interpretation, seq_delta in zip(fids, pos, attributions, delta):
                if record == current_subj:
                    interpretation = np.concatenate((interpretation, seq_interpretation.cpu().detach().numpy()), 1)
                    err_delta = np.concatenate((err_delta, np.expand_dims(seq_delta.cpu().detach().numpy(), 0)), 0)
                    rec_pos.append(b_pos[0])
                else:
                    if current_subj != '' and save_path is not None:
                        # Save interpretation as h5 files
                        output_filename = os.path.join(save_path, current_subj)
                        with h5py.File(output_filename, "w") as f:
                            # Save PSG
                            f.create_dataset("Interpretation", data=interpretation, dtype='f4')
                            f.create_dataset("Delta", data=err_delta, dtype='f4')
                            f.create_dataset("Position", data=np.array(rec_pos), dtype='i4')
                    # Create new array for new recording
                    current_subj = record
                    interpretation = seq_interpretation.cpu().detach().numpy()
                    err_delta = np.expand_dims(seq_delta.cpu().detach().numpy(), 0)
                    rec_pos = [b_pos[0]]

        # Save interpretation as h5 files
        output_filename = os.path.join(save_path, current_subj)
        with h5py.File(output_filename, "w") as f:
            # Save PSG
            f.create_dataset("Interpretation", data=interpretation, dtype='f4')
            f.create_dataset("Delta", data=err_delta, dtype='f4')
            f.create_dataset("Position", data=np.array(rec_pos), dtype='i4')

        return

    def occlusion_attribution(self, x):
        occlude_channel_sets = [[0, 1], [2, 3], [4], [5], [6], [7, 8, 9, 10], [11]]
        occlude_ref_style = ['same', 'same', 'same', 'same', 'same', 'same']
        occlude_window = 5 * 128

        y = self.best_network(x)

        y_occ_all = torch.zeros_like(x)

        for occ_set in occlude_channel_sets:
            for occ_start in range(0, 5 * 128 * 60, occlude_window):
                x_occ = x.detach().clone()
                x_occ[:, occ_set, occ_start:(occ_start + occlude_window)] = 0
                y_occ = self.best_network(x_occ)
                y_occ_all[:, occ_set, occ_start:(occ_start + occlude_window)] = y_occ.detach().clone().unsqueeze(
                    2).repeat(1, len(occ_set), occlude_window)

        attribution = y_occ_all - y.detach().clone().unsqueeze(2).repeat(1, len(occ_set), 128 * 60 * 5)
        return attribution, torch.zeros_like(x)

# Main code block that sets up the configuration, model, dataset, and starts the training process
if __name__ == '__main__':
    config = Config()  # Load the configuration settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set the computing device
    loss_fn = insomnia_loss(device, config.loss_func)  # Initialize the loss function
    model = BasicModel(config)  # Initialize the model
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # Setup the optimizer
    model.to(device)  # Move the model to the specified computing device
    dataset = "train"  # Specify the dataset mode (train, validation, etc.)
    ds = PSG_Dataset(config, dataset)  # Load the dataset
    dl = DataLoader(ds, shuffle=False, batch_size=8, num_workers=0, pin_memory=True, drop_last=True)  # Setup the DataLoader
    bar_data = tqdm(dl, total=len(dl), desc=f'Loss: {np.inf:.04f}')  # Progress bar for data loading
    batch = next(iter(bar_data))  # Load a single batch of data
    x, y = batch['data'].to(device), batch['label'].type(torch.LongTensor).to(device)  # Prepare the inputs and labels
    y_p = model(x)  # Forward pass through the model
    loss_out = loss_fn.loss(y_p, y)  # Compute the loss
