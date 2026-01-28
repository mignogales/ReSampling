"""
TimeSeriesDataModule with arbitrary resampling rate support.

Uses scipy.signal.resample for non-integer rates, enabling zero-shot
evaluation of models trained at one temporal resolution on degraded data.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy as np
from scipy import signal
from extras.timeseriesdataset import TimeSeriesDataset
import datetime
import pandas as pd


class TimeSeriesDataModuleResampled(pl.LightningDataModule):
    """
    DataModule supporting arbitrary resampling rates via scipy.signal.resample.
    
    The resample_rate parameter controls temporal resolution:
        - 1.0: Original resolution (no resampling)
        - 2.0: Half the samples (coarser resolution)
        - 0.5: Double the samples (finer resolution, interpolation)
    
    Resampling is applied ONLY to the test set to evaluate zero-shot robustness.
    Train/val sets remain at original resolution for scaler fitting consistency.
    """
    
    def __init__(self, 
                 data,
                 window_size: int,
                 batch_size: int = 32,
                 frequency: str = None,
                 forecast_horizon: int = None,
                 contain_missing_values: bool = False,
                 contain_equal_length: bool = False,
                 workers: int = 4,
                 splits: dict = {'val_len': 0.15, 'test_len': 0.15}, 
                 transform: dict = None,
                 resample_rate: float = 1.0,
                 change_effective_window: bool = True,
                 resample_method: str = 'fourier'):
        """
        Args:
            data: Raw data DataFrame with 'series_value' and 'start_timestamp' columns.
            window_size: Lookback window size (in original samples).
            batch_size: Batch size for DataLoaders.
            frequency: Data frequency ('minutely', '10_minutes', etc.).
            forecast_horizon: Number of steps to forecast.
            contain_missing_values: Whether data contains NaNs.
            contain_equal_length: Whether all series have equal length.
            workers: Number of DataLoader workers.
            splits: Dict with 'val_len' and 'test_len' proportions.
            transform: Dict with 'target' key containing a scaler.
            resample_rate: Resampling factor (>1 = downsample, <1 = upsample).
            resample_method: 'fourier' (scipy.signal.resample) or 'interp' (numpy.interp).
        """
        super().__init__()
        
        if isinstance(data, torch.Tensor):
            data = data.numpy()

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
        self.resample_rate = resample_rate
        self.resample_method = resample_method
        
        self.scaler = None
        
        # Adjusted window size for resampled data
        # Critical: window_size in resampled space = window_size / resample_rate

        if change_effective_window:
            self.effective_window_size = max(1, int(np.round(window_size / resample_rate)))
            self.effective_horizon = max(1, int(np.round(forecast_horizon / resample_rate)))
        else:
            self.effective_window_size = window_size
            self.effective_horizon = forecast_horizon

    def _resample_series(self, series: np.ndarray) -> np.ndarray:
        """
        Resample a single time series using scipy.signal.resample.
        
        This uses Fourier-based resampling which preserves frequency content
        better than simple decimation for non-integer rates.
        
        Args:
            series: 1D numpy array of time series values.
            
        Returns:
            Resampled series with length = len(series) / resample_rate.
        """
        if self.resample_rate == 1.0:
            return series
        
        original_length = len(series)
        new_length = max(1, int(np.round(original_length / self.resample_rate)))
        
        # Handle NaN values: interpolate before resampling, then restore NaN positions
        has_nans = np.isnan(series).any()
        if has_nans:
            nan_mask = np.isnan(series)
            # Linear interpolation for NaN handling
            valid_indices = np.where(~nan_mask)[0]
            if len(valid_indices) > 1:
                series_filled = np.interp(
                    np.arange(original_length),
                    valid_indices,
                    series[valid_indices]
                )
            else:
                series_filled = np.nan_to_num(series, nan=0.0)
        else:
            series_filled = series
        
        if self.resample_method == 'fourier':
            # Fourier-based resampling (optimal for preserving spectral content)
            resampled = signal.resample(series_filled, new_length)
        elif self.resample_method == 'interp':
            # Linear interpolation (faster, but less accurate for high-frequency content)
            x_old = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, new_length)
            resampled = np.interp(x_new, x_old, series_filled)
        else:
            raise ValueError(f"Unknown resample_method: {self.resample_method}")
        
        return resampled.astype(np.float32)

    def setup(self, stage=None):
        """
        Prepare train/val/test datasets.
        
        Key design decision: Scaler is fitted on ORIGINAL resolution training data,
        then applied to resampled test data. This simulates realistic deployment
        where preprocessing was calibrated on training distribution.
        """
        
        # Determine sampling rate in seconds
        if self.frequency == "10_minutes":
            sampling_rate = 600
        elif self.frequency == "minutely":
            sampling_rate = 60
        else:
            raise ValueError(f"Unknown frequency: {self.frequency}")

        # Generate timestamps (at original resolution)
        self.data['time_stamps'] = self.data.apply(
            lambda row: np.array([
                row['start_timestamp'] + datetime.timedelta(seconds=i * sampling_rate) 
                for i in range(len(row['series_value']))
            ]), 
            axis=1
        )

        # Compute split boundaries
        all_time_stamps = np.concatenate(self.data['time_stamps'].values)
        unique_time_stamps = np.unique(all_time_stamps)
        total_duration = len(unique_time_stamps)

        train_limits = int(total_duration * (1 - self.splits['val_len'] - self.splits['test_len']))
        val_limits = int(total_duration * (1 - self.splits['test_len']))
        test_limits = total_duration - 1

        # Prepare DataFrames for each split
        df_train = self.data.copy()
        df_val = self.data.copy()
        df_test = self.data.copy()

        for idx, row in self.data.iterrows():
            serie = np.array(row['series_value'], dtype=np.float32)
            time = np.array(row['time_stamps'])

            train_end = np.searchsorted(time, unique_time_stamps[train_limits], side='right')
            val_end = np.searchsorted(time, unique_time_stamps[val_limits], side='right')
            test_end = np.searchsorted(time, unique_time_stamps[test_limits], side='right')

            df_train.at[idx, 'series_value'] = serie[:train_end]
            df_val.at[idx, 'series_value'] = serie[train_end:val_end]
            df_test.at[idx, 'series_value'] = serie[val_end:test_end]

        # Fit scaler on ORIGINAL resolution training data
        train_raw = [np.array(row['series_value'], dtype=np.float32) for _, row in df_train.iterrows()]
        train_raw = np.concatenate(train_raw, axis=0).reshape(-1, 1)
        
        if 'target' in self.transform:
            self.scaler = self.transform['target']
            self.scaler.fit(train_raw)
            
            # Apply scaler to all data
            for idx, row in self.data.iterrows():
                serie = np.array(row['series_value'], dtype=np.float32).reshape(-1, 1)
                scaled_serie = self.scaler.transform(serie).flatten()
                self.data.at[idx, 'series_value'] = scaled_serie

        # Handle unequal series lengths
        equal_start_timestamps = len(set(self.data['start_timestamp'])) == 1
        equal_length = len(set(self.data['series_value'].map(len))) == 1
        
        if equal_start_timestamps and not equal_length:
            max_length = max(self.data['series_value'].map(len))
            for idx, row in self.data.iterrows():
                serie = np.array(row['series_value'], dtype=np.float32)
                padded_serie = np.pad(serie, (0, max_length - len(serie)), 
                                      mode='constant', constant_values=np.nan)
                self.data.at[idx, 'series_value'] = padded_serie
        elif not equal_start_timestamps:
            raise NotImplementedError("Currently only equal start timestamps are supported.")

        # Create splits with context overlap
        train_data = slice_df_series(self.data, 0, train_limits)
        val_data = slice_df_series(self.data, train_limits - self.window_size, val_limits)
        test_data = slice_df_series(self.data, val_limits - self.window_size, test_limits)

        # Apply resampling ONLY to test data
        if self.resample_rate != 1.0:
            test_data = self._resample_dataframe(test_data)
            print(f"[Resampling] Test data resampled at rate {self.resample_rate}x")
            print(f"[Resampling] Effective window: {self.effective_window_size}, horizon: {self.effective_horizon}")

        contain_missing_values = self.contain_missing_values or not equal_length

        # Instantiate datasets
        # Note: Train/Val use original window_size, Test uses effective_window_size
        self.train_dataset = TimeSeriesDataset(
            train_data, self.window_size, self.forecast_horizon, contain_missing_values
        )
        self.val_dataset = TimeSeriesDataset(
            val_data, self.window_size, self.forecast_horizon, contain_missing_values
        )
        self.test_dataset = TimeSeriesDataset(
            test_data, self.effective_window_size, self.effective_horizon, contain_missing_values
        )

    def _resample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply resampling to all series in a DataFrame."""
        df_resampled = df.copy()
        
        for idx, row in df_resampled.iterrows():
            serie = np.array(row['series_value'], dtype=np.float32)
            resampled_serie = self._resample_series(serie)
            df_resampled.at[idx, 'series_value'] = resampled_serie
            
        return df_resampled

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=self.workers, 
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=self.workers, 
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=self.workers, 
            persistent_workers=True
        )


def slice_df_series(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    """Slice each series in the DataFrame to [start_idx:end_idx]."""
    sliced_df = df.copy()
    for idx, row in sliced_df.iterrows():
        serie = np.array(row['series_value'], dtype=np.float32)
        sliced_df.at[idx, 'series_value'] = serie[start_idx:end_idx]
    return sliced_df
