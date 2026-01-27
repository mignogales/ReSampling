import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy as np
from extras.timeseriesdataset import TimeSeriesDataset
import datetime
import pandas as pd

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data,
                 window_size,
                 batch_size=32,
                 frequency=None,
                 donwsamples=0,
                 forecast_horizon=None,
                 contain_missing_values=False,
                 contain_equal_length=False,
                 workers=4,
                 splits={'val_len': 0.15, 'test_len': 0.15}, 
                 transform=None):
        """
        Args:
            data: Raw data (List, Numpy array, or Tensor). Shape: [N, Features] or [N]
            window_size: Lookback window size.
            batch_size: Batch size for loaders.
            splits: Dictionary defining validation and test proportions.
            transform: Dictionary with 'target' key containing a Scikit-Learn scaler.
        """
        super().__init__()
        # Ensure data is numpy for Scikit-Learn compatibility
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # Go through each series and create a TimeSeriesDataset
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size
        self.splits = splits
        self.transform = transform if transform is not None else {}
        self.frequency = frequency
        self.forecast_horizon = forecast_horizon
        self.contain_missing_values = contain_missing_values
        self.contain_equal_length = contain_equal_length
        self.workers = workers
        self.downsamples = donwsamples
        
        # Placeholder for the scaler to be accessed later
        self.scaler = None

    def setup(self, stage=None):
        
        if self.frequency == "10_minutes":
            sampling_rate = 600
        elif self.frequency == "minutely":
            sampling_rate = 60
        else:
            raise ValueError(f"Unknown frequency: {self.frequency}")

        # generate time stamps for each time sample when ds is 0. create one time serie per row in the dataframe
        if self.downsamples == 0:
            self.data['time_stamps'] = self.data.apply(lambda row: np.array([row['start_timestamp'] + datetime.timedelta(seconds=i * sampling_rate) for i in range(len(row['series_value']))]), axis=1)

        df_train = self.data.copy()
        df_val = self.data.copy()
        df_test = self.data.copy()
        # 1. Data Downsampling and Frequency Adjustment
        series_frequencies = {}
        for ds in range(self.downsamples + 1):

            # 1. Data Downsampling (if required)
            if ds != 0:
    
                sampling_rate = sampling_rate * 2

                for idx, row in self.data.iterrows():
                    raw_series = np.array(row['series_value'], dtype=np.float32)
                    # Downsample by taking every 2nd value
                    downsampled_series = downsample_serie(raw_series, factor=ds)
                    self.data.at[idx, 'series_value'] = downsampled_series

            if ds == 0:
                # define train, val, test durations based on the total time stamps length
                # total time stamp is the union of all series time stamps
                # define the min and max of each set by datetime
                all_time_stamps = np.concatenate(self.data['time_stamps'].values)
                unique_time_stamps = np.unique(all_time_stamps)
                total_duration = len(unique_time_stamps)

                train_limits =  int(total_duration * (1 - self.splits['val_len'] - self.splits['test_len']))
                val_limits = int(total_duration * (1 - self.splits['test_len']))
                test_limits = total_duration - 1

            # from each serie extract the corresponding number of samples for train, val, test. consider each serie could have different length
            for idx, row in self.data.iterrows():
                serie = np.array(row['series_value'], dtype=np.float32)
                time = np.array(row['time_stamps'])

                # find the end index for train based on time stamps
                train_end = np.searchsorted(time, unique_time_stamps[train_limits], side='right')
                val_end = np.searchsorted(time, unique_time_stamps[val_limits], side='right')
                test_end = np.searchsorted(time, unique_time_stamps[test_limits], side='right')

                # change in the df the series value to be the sliced version
                df_train.at[idx, 'series_value'] = serie[:train_end]
                df_val.at[idx, 'series_value'] = serie[train_end:val_end]
                df_test.at[idx, 'series_value'] = serie[val_end:test_end]
                # also update time stamps
                df_train.at[idx, 'time_stamps'] = time[:train_end]
                df_val.at[idx, 'time_stamps'] = time[train_end:val_end]
                df_test.at[idx, 'time_stamps'] = time[val_end:test_end]

            # Collect training data for scaler fitting
            train_raw = [np.array(row['series_value'], dtype=np.float32) for _, row in df_train.iterrows()]

            # flatten the list of arrays into a single tensor
            train_raw = np.concatenate(train_raw, axis=0).reshape(-1, 1)

            series_frequencies[ds] = sampling_rate
        
        # 2. Fit Scaler (ONLY on Training Data)
        if 'target' in self.transform:
            self.scaler = self.transform['target']
            self.scaler.fit(train_raw)
            # Transform the WHOLE dataset serie by serie
            for idx, row in self.data.iterrows():
                serie = np.array(row['series_value'], dtype=np.float32).reshape(-1, 1)
                scaled_serie = self.scaler.transform(serie).flatten()
                self.data.at[idx, 'series_value'] = scaled_serie
        else:
            data_processed = self.data

        # check if all series start at the same time (equal start timestamps)
        equal_start_timestamps = len(set(self.data['start_timestamp'])) == 1
        # check if all series have the same length
        equal_length = len(set(self.data['series_value'].map(lambda x: len(x)))) == 1
        if equal_start_timestamps and not equal_length:
            # pad the series to the same length with NaNs
            max_length = max(self.data['series_value'].map(lambda x: len(x)))
            for idx, row in self.data.iterrows():
                serie = np.array(row['series_value'], dtype=np.float32)
                padded_serie = np.pad(serie, (0, max_length - len(serie)), mode='constant', constant_values=np.nan)
                self.data.at[idx, 'series_value'] = padded_serie
        elif not equal_start_timestamps:
            raise NotImplementedError("Currently only equal start timestamps are supported.")

        # 3. Create Splits with "Context Overlap"
        # We need the previous 'window_size' steps to predict the first element of the next split.
        # 3. Create Splits with "Context Overlap"
        train_data = slice_df_series(self.data, 0, train_limits)

        val_data = slice_df_series(self.data, train_limits - self.window_size, val_limits)

        test_data = slice_df_series(self.data, val_limits - self.window_size, test_limits)

        contain_missing_values = self.contain_missing_values or not equal_length

        # 4. Instantiate Datasets
        self.train_dataset = TimeSeriesDataset(train_data, self.window_size, self.forecast_horizon, contain_missing_values)
        self.val_dataset = TimeSeriesDataset(val_data, self.window_size, self.forecast_horizon, contain_missing_values)
        self.test_dataset = TimeSeriesDataset(test_data, self.window_size, self.forecast_horizon, contain_missing_values)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.workers, persistent_workers=True)

def slice_df_series(df, start_idx, end_idx):
    
    train_data = df.copy()
    # slice each series to the train duration
    for idx, row in train_data.iterrows():
        serie = np.array(row['series_value'], dtype=np.float32)
        train_serie = serie[start_idx:end_idx]
        train_data.at[idx, 'series_value'] = train_serie

    return train_data


def downsample_serie(serie, factor=2, method='naive'):
    # Downsample by taking every 'factor' value
    if method == 'naive':
        return serie[::factor]
    elif method == 'average':
        return np.mean(serie.reshape(-1, 2**factor), axis=1)
    else:
        raise ValueError("Unknown downsampling method: {}".format(method))
    