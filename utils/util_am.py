# Import necessary modules from PyTorch
import torch
import torch.nn as nn

# Define the am_model class that inherits from nn.Module
class am_model(nn.Module):
    def __init__(self, net, in_size):
        super(am_model, self).__init__()  # Call to the constructor of the superclass (nn.Module)
        # Create a learnable parameter for the model input using nn.Parameter.
        # This parameter is initialized with random values and has the specified input size.
        self.am_data = nn.Parameter(torch.randn(in_size))
        # Store the network passed to this model. This network will be used in the forward pass.
        self.net = net
        # Iterate over all parameters in the network and set their requires_grad attribute to False.
        # This effectively freezes the parameters of the network, making them non-trainable.
        # The model will focus on optimizing the am_data parameter instead.
        for param in self.net.parameters():
            param.requires_grad = False

    # Define the forward pass of the model
    def forward(self):
        # Pass the learnable am_data parameter through the network.
        # The network's parameters are frozen, so am_data will be the only source of optimization.
        x = self.net(self.am_data)
        return x  # Return the output of the network
