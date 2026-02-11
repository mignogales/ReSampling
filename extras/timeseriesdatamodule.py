import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy as np
from extras.timeseriesdataset import TimeSeriesDataset
import datetime
from sklearn.preprocessing import StandardScaler
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
                 transform=None,
                 training_mode="normal"): # normal for typical training, teacher_forcing for teacher forcing training
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
        self.training_mode = training_mode
        
        # Placeholder for the scaler to be accessed later
        self.scaler = None

    def setup(self, stage=None):
        
        if self.frequency == "10_minutes":
            sampling_rate = 600
        elif self.frequency == "minutely":
            sampling_rate = 60
        elif self.frequency == "daily":
            sampling_rate = 86400
        else:
            raise ValueError(f"Unknown frequency: {self.frequency}")

        df_train = self.data.copy()

        # check how many series we have (len of key series_name), check if key series_type exists for later normalization. maybe the key does not exists, take that into account
        series_names = self.data['series_name'].unique()
        series_types = self.data['series_type'].unique() if 'series_type' in self.data.columns else ["only_type"]

        # 1. Determine Split Indices. we will split each series based on its length individually
        series_split_indices = {}
        for series_name in series_names:
            series_data = df_train[df_train['series_name'] == series_name]
            series_length = series_data['series_value'].map(len).max()
            series_split_indices[series_name] = {
                'train': (0, int(series_length * (1 - self.splits['val_len'] - self.splits['test_len']))),
                'val': (int(series_length * (1 - self.splits['val_len'] - self.splits['test_len'])),int(series_length * (1 - self.splits['test_len']))),
                'test': (int(series_length * (1 - self.splits['test_len'])), series_length - 1)
            }

        

        # total_duration = max(self.data['series_value'].map(lambda x: len(x)))

        # train_limits =  int(total_duration * (1 - self.splits['val_len'] - self.splits['test_len']))
        # val_limits = int(total_duration * (1 - self.splits['test_len']))
        # test_limits = total_duration - 1

        # # Collect training data for scaler fitting
        # train_raw = [np.array(row['series_value'], dtype=np.float32) for _, row in df_train.iterrows()]

        # # flatten the list of arrays into a single tensor
        # train_raw = np.concatenate(train_raw, axis=0).reshape(-1, 1)

        
        # 2. Fit Scaler (ONLY on Training Data), one for each series type if exists. but we will keep each time series separated for fitting
        self.scaler = {}
        for series_type in series_types:
            series_type_data = df_train[df_train['series_type'] == series_type] if 'series_type' in self.data.columns else df_train
            train_raw = []
            for _, row in series_type_data.iterrows():
                serie = np.array(row['series_value'], dtype=np.float32)
                train_limit = series_split_indices[row['series_name']]['train'][1]
                train_serie = serie[0:train_limit]
                train_raw.append(train_serie)
            train_raw = np.concatenate(train_raw, axis=0).reshape(-1, 1)

            scaler = StandardScaler()
            scaler.fit(train_raw)
            self.scaler[series_type] = scaler

        # Apply transformation to the entire dataset
        for idx, row in df_train.iterrows():
            serie = np.array(row['series_value'], dtype=np.float32).reshape(-1, 1)
            series_type = row['series_type'] if 'series_type' in self.data.columns else "only_type"
            scaler = self.scaler[series_type]
            transformed_serie = scaler.transform(serie).flatten()
            df_train.at[idx, 'series_value'] = transformed_serie

        # 3. Create Splits with "Context Overlap"
        # We need the previous 'window_size' steps to predict the first element of the next split.
        # 3. Create Splits with "Context Overlap"
        train_data = slice_df_series(df_train, series_split_indices, type='train')
        val_data = slice_df_series(df_train, series_split_indices, type='val')
        test_data = slice_df_series(df_train, series_split_indices, type='test')

        contain_missing_values = self.contain_missing_values

        # 4. Instantiate Datasets
        print("Creating TimeSeriesDataset instances...")

        self.train_dataset = TimeSeriesDataset(train_data, self.window_size, self.forecast_horizon, contain_missing_values, training_mode=self.training_mode)
        self.val_dataset = TimeSeriesDataset(val_data, self.window_size, self.forecast_horizon, contain_missing_values)
        self.test_dataset = TimeSeriesDataset(test_data, self.window_size, self.forecast_horizon, contain_missing_values)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.workers, persistent_workers=True)

def slice_df_series(df, series_split_indices, type='train'):
    # Slices each series in the dataframe according to the provided split indices
    # take into account that there will be several series in the dataframe
    sliced_data = []
    for idx, row in df.iterrows():
        serie = np.array(row['series_value'], dtype=np.float32)
        series_name = row['series_name']
        start_idx, end_idx = series_split_indices[series_name][type]
        sliced_serie = serie[start_idx:end_idx]
        new_row = row.copy()
        new_row['series_value'] = sliced_serie
        sliced_data.append(new_row)
    return pd.DataFrame(sliced_data)


def downsample_serie(serie, factor=2, method='naive'):
    # Downsample by taking every 'factor' value
    if method == 'naive':
        return serie[::factor]
    elif method == 'average':
        return np.mean(serie.reshape(-1, 2**factor), axis=1)
    else:
        raise ValueError("Unknown downsampling method: {}".format(method))
    