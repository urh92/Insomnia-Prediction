import argparse

# Define the str2bool function that converts a string to a boolean value
def str2bool(v):
    # Check if the input value is already a boolean type
    if isinstance(v, bool):
        return v  # If so, just return the value as is

    # Convert the input string to lowercase and check if it represents a 'True' value
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True  # These string values are considered to represent 'True'

    # Convert the input string to lowercase and check if it represents a 'False' value
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False  # These string values are considered to represent 'False'

    # If the input string does not match any of the above cases, it's not a valid boolean representation
    else:
        # Raise an error specific to argparse to indicate that a boolean value was expected but not provided
        raise argparse.ArgumentTypeError('Boolean value expected.')
