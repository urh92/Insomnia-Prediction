# Import necessary libraries and modules
from channel_names import * 
from fractions import Fraction 
from scipy import interpolate 
from scipy.signal import resample_poly, iirfilter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to edit the first line of a file, presumably to clean or modify the header
def edit_file(fid):
    lines = fid.readlines()  # Read all lines from the file
    header = bytearray(lines[0])  # Convert the first line (header) into a mutable bytearray
    # Replace characters in positions 8 to 88 with spaces (for anonymization or formatting reasons)
    header[8:88] = b'                                                                                '
    # Replace non-standard degree symbol with a question mark
    header = header.replace(b'\xb0', b'?')
    header = bytes(header)  # Convert bytearray back to bytes (immutable)
    lines[0] = header  # Update the first line with the modified header

    return lines  # Return the modified lines

# Function to correct units of measurement to a consistent format
def correct_units(unit_string):
    # Convert unit string to a multiplication factor
    if unit_string == 'uV':
        g = 0.001  # Microvolts to volts
    elif unit_string == 'V':
        g = 1.0  # Volts remain unchanged
    else:
        g = 1.0  # Default case, no change

    return g  # Return the conversion factor

# Function to resample a signal to a desired frequency
def resample(sig, channel, fs, method):
    # Resample signal using polynomial method or linear interpolation
    if method == 'poly':
        # Calculate resampling fraction with a maximum denominator of 100
        resample_frac = Fraction(des_fs[channel] / fs).limit_denominator(100)
        # Resample using polynomial method
        sig = resample_poly(sig, resample_frac.numerator, resample_frac.denominator)
    elif method == 'linear':
        # Time vector for original signal
        t = np.arange(0, len(sig) * (1 / fs), 1 / fs)
        # Interpolate signal
        resample_f = interpolate.interp1d(t, sig, bounds_error=False, fill_value='extrapolate')
        # Time vector for resampled signal
        t_new = np.arange(0, len(sig) * (1 / fs), 1 / des_fs[channel])
        sig = resample_f(t_new)
    return sig  # Return resampled signal

# Function to create a high-pass filter specification
def psg_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    # Determine filter type and normalize cutoff frequencies
    if isinstance(cutoff, list):
        normal_cutoff = [x / nyq for x in cutoff]
        btype = 'bandpass'
    else:
        normal_cutoff = cutoff / nyq
        btype = 'highpass'
    # Create a second-order sections filter
    sos = iirfilter(order, normal_cutoff, rp=1, rs=40, btype=btype, analog=False, output='sos', ftype='ellip')

    return sos  # Return the filter

# Function to apply the high-pass filter to data
def psg_highpass_filter(data, cutoff, fs, order=5):
    sos = psg_highpass(cutoff, fs, order=order)  # Get the filter
    y = sosfiltfilt(sos, data)  # Filter the data

    return y  # Return the filtered data

# Function to rescale data using various methods
def rescale(x, mode):
    eps = 1e-10  # Small epsilon to avoid division by zero
    # Rescale data based on the mode specified
    if mode == 'hard':
        x_s = 2 * (x - min(x)) / (max(x) - min(x) + eps) - 1
    elif mode == 'soft':
        q5 = np.percentile(x, 5)
        q95 = np.percentile(x, 95)
        x_s = 2 * (x - q5) / (q95 - q5 + eps) - 1
        return x_s, q5, q95
    elif mode == 'standardize':
        scaler = StandardScaler()
        x_s = scaler.fit_transform(x.reshape(-1, 1)).flatten()
    elif mode == 'osat':
        if max(x) > 1.0:
            x = np.array(x / 100.0)  # Assuming input might need to be scaled down
        x[x < 0.6] = 0.6  # Thresholding
        x_s = 2 * (x - 0.6) / (1.0 - 0.6) - 1
    else:
        x_s = x  # No rescaling applied
    return x_s, 0, 0  # Return the rescaled data and dummy values for consistency
