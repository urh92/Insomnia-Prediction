# Import necessary libraries
import os
import pyedflib
from signal_processing import *
import pandas as pd
import h5py
import numpy as np
import glob
from channel_names import *  # Assumes channel_names.py contains mappings for channel aliases
import shutil

# Define a class for handling Polysomnography (PSG) data
class PSG:
    def __init__(self, filepath, labels_path=None, arousal_path=None, stage_path=None, h5_path=None, num_missing_max=2):
        # Initialize the PSG object with paths and parameters
        self.filepath = filepath  # Path to EDF files
        self.labels_path = labels_path  # Path to labels file (optional)
        self.arousal_path = arousal_path  # Path to arousal files (optional)
        self.stage_path = stage_path  # Path to sleep stage files (optional)
        self.h5_path = h5_path  # Path to output HDF5 files (optional)
        # Predefined list of channels to look for in the EDF files
        self.channels = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'LEOG', 'REOG', 'Chin', 'ECG']
        self.num_missing_max = num_missing_max  # Maximum allowed number of missing channels

        # Find all EDF files in the given path
        self.files = [file for file in os.listdir(self.filepath) if file.lower().endswith(".EDF".lower())]
        # Load labels from Excel file if provided
        self.labels = pd.read_excel(self.labels_path) if labels_path else None
        # Find arousal and sleep stage files if paths are provided
        self.arousals = [ar for ar in os.listdir(self.arousal_path) if ar.lower().endswith(".TXT".lower())] if self.arousal_path else []
        self.stages = [ss for ss in os.listdir(self.stage_path) if ss.lower().endswith("TSV".lower())] if self.stage_path else []

    def set_files(self):
        # Filter files based on availability of corresponding arousal and stage files
        if not self.arousals or self.stages:
            raise Exception("Missing path for arousal and/or sleep stage files")
        self.files = sorted([f for f in self.files if int(f[:-4]) in self.labels["Subjects"].unique() and
                             (f[:-4] + '.txt') in self.arousals and (f[:-4] + '.tsv') in self.stages])

    def get_channel_labels(self, channel):
        # Retrieve and return unique combinations of channel labels present across all files
        label_combinations, missing = [], []
        for file in self.files:
            f = pyedflib.EdfReader(os.path.join(self.filepath, file))
            signal_labels = f.getSignalLabels()
            # Combine all possible aliases for the given channel
            channel_labels = channel_alias[channel] + unref_channel_alias[channel] + ref_channel_alias[channel]
            channel_labels = list(dict.fromkeys(channel_labels))  # Remove duplicates
            labels_in_file = sorted([c for c in channel_labels if c in signal_labels])
            if not labels_in_file:
                missing.append(file)  # Track files where the channel is missing
                continue
            if labels_in_file not in label_combinations:
                label_combinations.append(labels_in_file)
        return label_combinations, missing

    def get_channels_labels(self, channels):
        # Wrapper to get channel labels for multiple channels
        channel_dict = {}
        for channel in channels:
            channel_dict[channel] = self.get_channel_labels(channel)
        return channel_dict

    def get_channel_units(self, channel):
        # Retrieve and return the units for a given channel across all files
        units = []
        for file in self.files:
            f = pyedflib.EdfReader(os.path.join(self.filepath, file))
            signal_labels = f.getSignalLabels()
            channel_labels = channel_alias[channel] + unref_channel_alias[channel] + ref_channel_alias[channel]
            if any([x in channel_labels for x in signal_labels]):
                channel_idx = signal_labels.index(next(filter(lambda i: i in channel_labels, signal_labels)))
                unit = f.getPhysicalDimension(channel_idx)
                if unit not in units:
                    units.append(unit)
        return units

    def get_channels_units(self, channels):
        # Wrapper to get units for multiple channels
        channel_dict = {}
        for channel in channels:
            channel_dict[channel] = self.get_channel_units(channel)
        return channel_dict

    def get_files_missing_channels(self, path_channel_labels):
        # Identify and return files missing specific channels, based on a provided list
        channel_dict = {}
        df_labels = pd.read_csv(path_channel_labels)
        for channel in df_labels['Channels']:
            missing = eval(df_labels[df_labels['Channels'] == channel]['Missing'].item())
            all_labels = []
            for file in missing:
                f = pyedflib.EdfReader(os.path.join(self.filepath, file))
                labels = f.getSignalLabels()
                labels = [l for l in labels if l not in all_labels]
                all_labels += labels
                f.close()
            channel_dict[channel] = all_labels
        return channel_dict

    def get_extra_files(self):
        # Find and return files that are considered extra based on naming convention
        extra_files = []
        for file in self.files:
            extra = glob.glob(os.path.join(self.filepath, file[:-4] + '*_[0-9].*'))
            if extra:
                extra_files.append(file)
        return extra_files

    def get_missing_predictions(self, predictions, new_folder=None):
        # Identify files missing arousal or stage predictions and optionally copy them to a new folder
        if predictions == 'arousals':
            prediction_files = [p_file[:-4] for p_file in self.arousals]
        elif predictions == 'stages':
            prediction_files = [p_file[:-4] for p_file in self.stages]
        diff_files = [file for file in self.files if file[:-4] not in prediction_files]
        if not new_folder:
            return diff_files
        else:
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            for diff_file in diff_files:
                shutil.copy(os.path.join(self.filepath, diff_file), os.path.join(new_folder, diff_file))

    def fix_header(self, file, output_path=None):
        # Attempt to fix the header of an EDF file and return whether it was successful
        edf_file_path = os.path.join(self.filepath, file)
        if output_path:
            output_file_path = os.path.join(output_path, file)
            if os.path.isfile(output_file_path) and self.check_fix(output_file_path):
                print("{}: Header already works and stored in output directory".format(file))
                return True
        else:
            if self.check_fix(edf_file_path):
                print("{}: Header already works".format(file))
                return True
            else:
                output_file_path = edf_file_path

        print("Fixing header of file: {}".format(file))
        with open(edf_file_path, 'rb') as f:
            new_file = edit_file(f)  # Assumes edit_file function is defined elsewhere

        with open(output_file_path, "wb") as f:
            f.writelines(new_file)

        return self.check_fix(output_file_path)

    def load_edf(self, file, channels):
        # Load EDF file, process signals based on provided channels, and return structured data
        # This includes reading signals, resampling, filtering, scaling, and appending arousal and sleep stage data
        edf_path = os.path.join(self.filepath, file)
        f = pyedflib.EdfReader(edf_path)
        channel_labels = f.getSignalLabels()
        fss = f.getSampleFrequencies()
        x = []
        q_low, q_high = [], []
        for channel in channels:
            if any([x in channel_alias[channel] for x in channel_labels]):
                channel_idx = channel_labels.index(next(filter(lambda i: i in channel_alias[channel], channel_labels)))
                g = correct_units(f.getPhysicalDimension(channel_idx))
                sig = g * f.readSignal(channel_idx)

            elif any([x in unref_channel_alias[channel] for x in channel_labels]) and any(
                    [x in ref_channel_alias[channel] for x in channel_labels]):
                channel_idx = channel_labels.index(
                    next(filter(lambda i: i in unref_channel_alias[channel], channel_labels)))
                ref_idx = channel_labels.index(next(filter(lambda i: i in ref_channel_alias[channel], channel_labels)))
                # Gain factor
                g = correct_units(f.getPhysicalDimension(channel_idx))
                g_ref = correct_units(f.getPhysicalDimension(ref_idx))
                # Assuming fs for signal and reference is identical
                sig = g * f.readSignal(channel_idx) - g_ref * f.readSignal(ref_idx)

            elif any([x in unref_channel_alias[channel] for x in channel_labels]):
                channel_idx = channel_labels.index(
                    next(filter(lambda i: i in unref_channel_alias[channel], channel_labels)))
                g = correct_units(f.getPhysicalDimension(channel_idx))
                # Read signal
                sig = g * f.readSignal(channel_idx)

            elif any([x in ref_channel_alias[channel] for x in channel_labels]):
                channel_idx = channel_labels.index(
                    next(filter(lambda i: i in ref_channel_alias[channel], channel_labels)))
                g = correct_units(f.getPhysicalDimension(channel_idx))
                # Read signal
                sig = g * f.readSignal(channel_idx)

            else:
                sig = []

            # If not empty
            if len(sig) != 0:
                # Resampling
                fs = fss[channel_idx]
                if fs != des_fs[channel]:
                    resample_method = 'linear' if channel == 'OSat' else 'poly'
                    sig = resample(sig, channel, fs, resample_method)

                # Filter signals
                if hp_fs[channel] != 0:
                    sig_filtered = psg_highpass_filter(sig, hp_fs[channel], des_fs[channel], order=16)
                else:
                    sig_filtered = sig
                # Scale signal
                scale_method = 'osat' if channel == 'OSat' else 'soft'
                sig_scaled, q5, q95 = rescale(sig_filtered, scale_method)
            else:
                sig_scaled, q5, q95 = sig, 0, 0

            x.append(sig_scaled)
            q_low.append(q5)
            q_high.append(q95)

        # Replace empty with zeros
        N = max([len(s) for s in x])
        for i, channel in enumerate(channels):
            if len(x[i]) == 0:
                x[i] = np.zeros(N)
            elif len(x[i]) != N:
                x[i] = np.append(x[i], np.zeros(N - len(x[i])))

        f.close()

        arousal_file_path = os.path.join(self.arousal_path, file[:-4] + '.txt')
        if not arousal_file_path:
            raise Exception("No arousal file for subject")
        arousals = self.load_arousals(arousal_file_path)
        arousals = np.repeat(arousals, 128)
        d = len(x[i])-len(arousals)
        if d > 0:
            arousals = np.concatenate((arousals, np.ones(d)))
        elif d < 0:
            print('Mismatch arousals: {}'.format(file))
        x.append(arousals)
        channel_ar = 'Arousals'
        all_channels = channels + [channel_ar]
        des_fs[channel_ar] = 128
        q_low.append(0)
        q_high.append(1)

        stage_file_path = os.path.join(self.stage_path, file[:-4] + '.tsv')
        if not stage_file_path:
            raise Exception("No sleep stage file for subject")
        stages = self.load_stages(stage_file_path)
        stages = np.repeat(stages, 128)
        d = len(x[i]) - len(stages)
        if d > 0:
            stages = np.concatenate((stages, np.zeros(d)))
        elif d < 0:
            print('Mismatch sleep stages: {}'.format(file))
        x.append(stages)
        channel_st = 'Stages'
        all_channels += [channel_st]
        des_fs[channel_st] = 128
        q_low.append(0)
        q_high.append(4)

        data = {'x': x, 'fs': [des_fs[x] for x in all_channels], 'channels': all_channels, 'q_low': q_low, 'q_high': q_high}
        return data

    def load_arousals(self, file):
        # Load arousal data from a file
        with open(os.path.join(self.arousal_path, file)) as f:
            data = [line for line in f.readlines()]
        arousals = np.array(eval(data[0][:-1]))
        return arousals

    def load_stages(self, file):
        # Load sleep stage data from a TSV file and return structured data
        stage_dict = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
        df = pd.read_csv(os.path.join(self.stage_path, file), sep='\t', header=0)
        df['init_sec'] = df['init_sec'].astype(int)
        df['duration_sec'] = df['duration_sec'].astype(int)
        sleep_stages = np.ones(df.iloc[-1]['init_sec'] + df.iloc[-1]['duration_sec']) * np.nan
        for i in range(len(df)):
            stage = df.iloc[i]['stage']
            start = df.iloc[i]['init_sec']
            dur = df.iloc[i]['duration_sec']
            sleep_stages[start:start + dur] = stage_dict[stage]
        return sleep_stages

    def load_labels(self, file):
        # Load insomnia label for a given subject from an Excel file
        subject_id = file[:-4]
        df = pd.read_excel(self.labels_path)
        label = df[df["Subjects"] == int(subject_id)]["Insomnia"].item()
        return label

    def write_2h5(self, file):
        # Process and save data to an HDF5 file
        data = self.load_edf(file, self.channels)
        extra_files = glob.glob(os.path.join(self.filepath, file[:-4] + '*_[0-9].*'))
        if extra_files:
            for extra_file in extra_files:
                data1 = self.load_edf(extra_file[-12:], self.channels)
                for i in range(len(data['x'])):
                    data['x'][i] = np.concatenate((data['x'][i], data1['x'][i]))
        if data['x'][0].shape[0] < data['fs'][0]*60*60*3:
            return
        sig = np.array(data['x'])
        channels = data['channels']
        q_low = data['q_low']
        q_high = data['q_high']
        label = self.load_labels(file)

        # If signals are nan, then dont save and return
        if (sig != sig).any():
            return

        # If too many signals are missing, then dont save and return
        if np.sum(np.count_nonzero(sig, 1) == 0) > self.num_missing_max:
            return

        output_file = os.path.join(self.h5_path, file[:-4]) + '.hdf5'
        with h5py.File(output_file, "w") as f:
            # Save PSG
            f.create_dataset("PSG", data=sig, dtype='f4', chunks=(len(channels), 128 * 60 * 5))
            # Save labels
            f.attrs['label'] = float(label)
            # Save data quantiles
            f.attrs['q_low'] = q_low
            f.attrs['q_high'] = q_high
        return

    def write_all_2h5(self):
        # Wrapper to process and save all files to HDF5
        for i in range(len(self.files)):
            file = self.files[i]
            h5_file = os.path.join(self.h5_path, file[:-4]) + '.hdf5'
            if not os.path.exists(h5_file):
                self.write_2h5(file)

    def delete_h5(self, file):
        # Delete a specific HDF5 file
        h5_file_path = os.path.join(self.h5_path, file[:-4] + '.hdf5')
        if os.path.exists(h5_file_path):
            os.remove(h5_file_path)

    def delete_extra_h5(self):
        # Delete HDF5 files deemed extra
        for extra_file in self.extra_files:
            self.delete_h5(extra_file)

    def check_fix(self, file):
        # Check if an EDF file can be opened without error, indicating a successful header fix
        try:
            f = pyedflib.EdfReader(file)
        except OSError:
            f = None
        return f

    def fix_all(self, output_path, n_fixed_path):
        # Attempt to fix all EDF files and record the names of files that couldn't be fixed
        n_fixed = []
        for fi in self.files:
            fixed = self.fix_header(fi, output_path)
            if not fixed:
                n_fixed.append(fi)
        if n_fixed:
            with open(os.path.join(n_fixed_path, 'n_fixed.txt'), 'w') as fp:
                for item in n_fixed:
                    fp.write("%s\n" % item)

# Example usage of the PSG class
if __name__ == '__main__':
    path = "C:/Users/UmaerHANIF/Documents/Pandore/EDFs"
    self = PSG(path)
    channels = self.get_channels_labels(channels=['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'LEOG', 'REOG'])
    df = pd.DataFrame.from_dict(channels, 'index', columns=['Combinations', 'Missing'])
    df = df.reset_index().rename(columns={'index': 'Channels'})
    df.to_csv('C:/Users/UmaerHANIF/Documents/Pandore/channel_labels.csv')
