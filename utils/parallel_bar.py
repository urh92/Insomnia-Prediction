# Import necessary modules
from joblib import Parallel, delayed
from tqdm import tqdm

# Define a dictionary mapping progress bar types to their corresponding functions
all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),  # For 'tqdm', use the tqdm function to create a progress bar
    'False': lambda args: iter,  # For 'False', simply use the iter function to create an iterator without a progress bar
    'None': lambda args: iter  # For 'None', also use iter indicating no progress bar is desired
}

# Define the ParallelExecutor function to configure and return a parallel execution function
def ParallelExecutor(use_bar='tqdm', **joblib_args):
    # Define the aprun function which configures and executes the parallel operation
    def aprun(bar=use_bar, **tq_args):
        # Define a temporary function to apply the operation iteratively with optional progress bar
        def tmp(op_iter):
            # Check if the specified bar type is supported and get the corresponding function
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                # If the bar type is not supported, raise a ValueError
                raise ValueError('Value %s not supported as bar type' % bar)
            # Execute the operation in parallel using joblib, applying the progress bar function if applicable
            return Parallel(**joblib_args)(bar_func(op_iter))
        # Return the temporary function
        return tmp
    # Return the aprun function
    return aprun
