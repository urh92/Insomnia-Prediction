# Import necessary libraries and modules
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

# Define a base model class which other models will inherit from
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the nn.Module superclass

    def forward(self):
        # This method is supposed to be overridden by all subclasses
        raise NotImplementedError

    # Method to print a summary of the model similar to what you get in Keras
    def summary(self, input_size, device, batch_size=-1):
        # Function to register a forward hook for each module
        def register_hook(module):
            # The hook function to gather information about layers
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]  # Get the class name of the module
                module_idx = len(summary)  # Index of the module
                m_key = '%s-%i' % (class_name, module_idx + 1)  # Create a unique key for the module
                summary[m_key] = OrderedDict()  # Initialize an ordered dict for the module in the summary
                summary[m_key]['input_shape'] = list(input[0].size())  # Record the input shape
                summary[m_key]['input_shape'][0] = batch_size  # Adjust the batch size in the input shape
                # Handle the output shapes, considering different layer types and scenarios (e.g., tuples of outputs)
                if isinstance(output, (list, tuple)):
                    if class_name in ('GRU', 'LSTM', 'RNN'):
                        summary[m_key]['output_shape'] = [batch_size] + list(output[0].size())[1:]
                    else:
                        summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = batch_size
                # Record whether the layer's parameters are trainable
                summary[m_key]['trainable'] = any(p.requires_grad for p in module.parameters())
                # Count the number of trainable parameters
                params = np.sum([np.prod(list(p.size())) for p in module.parameters() if p.requires_grad])
                summary[m_key]['nb_params'] = int(params)

            # Register the hook if the module is not a Sequential or ModuleList container,
            # and not the base model itself
            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != self:
                hooks.append(module.register_forward_hook(hook))

        # Ensure the specified device is valid
        assert device.type in ('cuda', 'cpu'), "Input device is not valid, please specify 'cuda' or 'cpu'"
        dtype = torch.cuda.FloatTensor if device.type == 'cuda' and torch.cuda.is_available() else torch.FloatTensor
        # Adjust input size format for consistency
        if isinstance(input_size, tuple):
            input_size = [input_size]
        # Generate a random tensor as input to the model for the summary
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        # Initialize summary dictionary and list of hooks
        summary = OrderedDict()
        hooks = []
        # Apply the hook registration function to all modules
        self.apply(register_hook)
        # Perform a forward pass to trigger the hooks
        self(*x)
        # Remove hooks after use
        for h in hooks:
            h.remove()

        # Printing the summary
        print('----------------------------------------------------------------')
        # Header line
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        print(line_new)
        print('================================================================')
        total_params = 0
        total_output = 0
        trainable_params = 0
        # Iterate through each layer in the summary
        for layer in summary:
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), '{0:,}'.format(summary[layer]['nb_params']))
            total_params += summary[layer]['nb_params']
            # Calculate total output size
            if any(isinstance(el, list) for el in summary[layer]['output_shape']):
                for list_out in summary[layer]['output_shape']:
                    total_output += np.prod(list_out, dtype=np.int64)
            else:
                total_output += np.prod(summary[layer]['output_shape'], dtype=np.int64)
            # Calculate trainable parameters
            if summary[layer]['trainable']:
                trainable_params += summary[layer]['nb_params']
            print(line_new)
        # Calculate and print the total sizes of inputs, outputs, parameters, and the estimated total size
        total_input_size = abs(np.prod(input_size) * batch_size * 4 / 1073741824)
        total_output_size = abs(2 * total_output * 4 / 1073741824)
        total_params_size = abs(total_params * 4 / 1073741824)
        total_size = total_params_size + total_output_size + total_input_size
        print('================================================================')
        print('Total params: {0:,}'.format(total_params))
        print('Trainable params: {0:,}'.format(trainable_params))
        print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        print('----------------------------------------------------------------')
        print('Input size (GB): %0.2f' % total_input_size)
        print('Forward/backward pass size (GB): %0.2f' % total_output_size)
        print('Params size (GB): %0.2f' % total_params_size)
        print('Estimated Total Size (GB): %0.2f' % total_size)
        print('----------------------------------------------------------------')

    # A method to debug the model by printing out the sizes of inputs and outputs
    def debug_model(self, input_size, device, cond_size=False):
        if cond_size or cond_size == 0:
            self.summary([input_size[1:], (cond_size,)], device, input_size[0])
            z = torch.rand((input_size[0], cond_size)).to(device)
        else:
            self.summary(input_size[1:], device, input_size[0])
        X = torch.rand(input_size).to(device)
        print('Input size: ', X.size())
        time_start = time.time()
        # Perform a forward pass
        if cond_size or cond_size == 0:
            out = self(X, z)
        else:
            out = self(X)
        print('Batch time: {:.3f}'.format(time.time() - time_start))
        # Print out the sizes of output tensors
        for k, v in out.items():
            print('Key: ', k)
            print('Output size: ', v.size())
