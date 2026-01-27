import torch
from torch.utils.data import Dataset
import numpy as np
from extras.data_loader import convert_tsf_to_dataframe
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data_df, 
                 input_len, 
                 pred_len, 
                 contain_missing_values=False,
                 cols_to_use='series_value'):
        """
        Args:
            data_path (str): Path to the .tsf file.
            input_len (int): Length of the input window (history).
            pred_len (int): Length of the prediction horizon (target).
            cols_to_use (str): The column name in the DF containing the series data.
            normalize (bool): Whether to standardize data (Z-score).
        """
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_window = input_len + pred_len
        self.contain_missing_values = contain_missing_values
        
        # 1. Load Data
        # its already a dataframe!
        self.loaded_data = data_df

        # 2. Pre-process and Build Index Map
        # We need a list of tuples: (series_idx, start_time_idx)
        # This maps a global dataset index (0 to N) to a specific window in specific series.
        self.samples_map = []
        
        # We iterate through every time series in the dataframe
        # Assuming 'series_value' holds the list/array of values for that series
        self.all_series = []
        
        for idx, row in self.loaded_data.iterrows():
            raw_series = np.array(row[cols_to_use], dtype=np.float32)
            
            # Optional: Handle missing values (NaN) here if needed
            if self.contain_missing_values:
                # Simple forward fill example (pandas logic on numpy array)
                mask = np.isnan(raw_series)
                if mask.any():
                     # processing logic here, e.g., interpolation
                    pass 

            self.all_series.append(raw_series)
            
            # Calculate how many valid windows exist in this specific series
            series_len = len(raw_series)
            num_windows = series_len - self.total_window + 1
            
            if num_windows > 0:
                # We store (series_index, start_index) for every valid window
                # This expands the dataset to be the sum of all valid windows across all series
                for t in range(num_windows):
                    self.samples_map.append((idx, t))
        
        print(f"Dataset created. Total samples: {len(self.samples_map)}")

    def __len__(self):
        return len(self.samples_map)

    def __getitem__(self, idx):
        # 1. Retrieve the metadata for this specific sample index
        series_idx, start_t = self.samples_map[idx]
        
        # 2. Retrieve the full series
        series_data = self.all_series[series_idx]
        
        # 3. Slice the window
        end_input = start_t + self.input_len
        end_target = end_input + self.pred_len

        x = series_data[start_t : end_input]
        y = series_data[end_input : end_target]
        
        # 4. Return as tensors
        # Adding a dimension for features: (Seq_Len, Features)
        # If univariate, Features = 1.
        return {
            'x': torch.tensor(x).unsqueeze(-1),
            'y': torch.tensor(y).unsqueeze(-1),
            # 'mask': None  # Placeholder for mask if needed
        }




    def get(self, item):

        # check if item is scalar or vector
        ndim = item.ndim if isinstance(item, Tensor) else 0
        if ndim == 0:  # get a single item
            sample = Data(pattern=self.batch_patterns)
        elif ndim == 1:  # get batch of items
            pattern = {
                name: ('b ' + pattern) if 't' in pattern else pattern
                for name, pattern in self.batch_patterns.items()
            }
            sample = StaticBatch(pattern=pattern, size=item.size(0))
        else:
            raise RuntimeError(f"Too many dimensions for index ({ndim}).")

        # get input synchronized with window
        if self.window > 0:
            wdw_idxs = self.get_window_indices(item)
            self._add_to_sample(sample, WINDOW, 'input', time_index=wdw_idxs)
            self._add_to_sample(sample, WINDOW, 'target', time_index=wdw_idxs)
            self._add_to_sample(sample,
                                WINDOW,
                                'auxiliary',
                                time_index=wdw_idxs)

        # get input synchronized with horizon
        hrz_idxs = self.get_horizon_indices(item)
        self._add_to_sample(sample, HORIZON, 'input', time_index=hrz_idxs)
        self._add_to_sample(sample, HORIZON, 'target', time_index=hrz_idxs)
        self._add_to_sample(sample, HORIZON, 'auxiliary', time_index=hrz_idxs)

        # get static data
        self._add_to_sample(sample, STATIC, 'input')
        self._add_to_sample(sample, STATIC, 'target')
        self._add_to_sample(sample, STATIC, 'auxiliary')

        # get connectivity
        if self.edge_index is not None:
            sample.input['edge_index'] = self.edge_index
            if self.edge_weight is not None:
                sample.input['edge_weight'] = self.edge_weight

        return sample