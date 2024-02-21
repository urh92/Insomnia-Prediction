# Import necessary libraries
import json 
import numpy as np

# Define a custom encoder that extends json.JSONEncoder
class NumpyEncoder(json.JSONEncoder):
    # Override the default method to handle numpy data types
    def default(self, obj):
        # Check if the object is an instance of numpy integer types
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)  # Convert numpy integers to Python int
        # Check if the object is an instance of numpy float types
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)  # Convert numpy floats to Python float
        # Check if the object is a numpy array
        elif isinstance(obj,(np.ndarray,)):  # This is the fix for handling numpy arrays
            return obj.tolist()  # Convert numpy arrays to Python lists
        # For all other data types, use the default JSONEncoder behavior
        return json.JSONEncoder.default(self, obj)
