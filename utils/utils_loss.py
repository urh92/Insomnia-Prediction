# Import necessary libraries
import torch
import torch.nn as nn

# Define a custom loss class for insomnia prediction
class insomnia_loss(nn.Module):
    # Initialization method for the insomnia_loss class
    def __init__(self, device, loss_method='cross-entropy', pos_weight=None, gamma_cov=0.01):
        super(insomnia_loss, self).__init__()  # Initialize the superclass (nn.Module)
        self.eps = 1e-10  # A small epsilon value to prevent division by zero or log(0)
        # Define the loss function to use based on the method specified, default is BCEWithLogitsLoss
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.device = device  # The device (CPU or GPU) where computations will be performed
        self.gamma_cov = gamma_cov  # A hyperparameter for potential future use in customizing the loss
        self.loss_method = loss_method  # The method of loss computation, default is cross-entropy

    # Forward pass of the loss computation
    def forward(self, y, t):
        # Compute the loss for insomnia prediction
        # y is the predicted values, t[:, 0] extracts the true labels for insomnia from the target tensor
        loss_insomnia = self.loss(y, t[:, 0])
        out = loss_insomnia
        return out
