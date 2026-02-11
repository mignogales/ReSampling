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
                 cols_to_use='series_value',
                 training_mode="normal"):
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
        self.training_mode = training_mode  # "normal" or "teacher_forcing"
        
        # 1. Load Data
        # its already a dataframe!
        self.loaded_data = data_df

        # 2. Pre-process and Build Index Map
        # We need a list of tuples: (series_idx, start_time_idx)
        # This maps a global dataset index (0 to N) to a specific window in specific series.
        self.samples_map = []
        
        # get all samples applying windowing. keep series separated. a continuous index has to be computed anyway, taken into account the different series lengths.
        self.all_series = []
        for series_idx, (_, row) in enumerate(self.loaded_data.iterrows()):
            series_data = np.array(row[cols_to_use], dtype=np.float32)
            series_length = len(series_data)
            self.all_series.append(series_data)
            
            # Determine valid start indices for windows in this series
            max_start_idx = series_length - self.total_window + 1
            for start_t in range(max_start_idx):
                self.samples_map.append((series_idx, start_t))        


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

        if self.training_mode == "normal":
            x = series_data[start_t : end_input]
            y = series_data[end_input : end_target]
        elif self.training_mode == "teacher_forcing":
            x = series_data[start_t : end_target - 1]  # input_len + pred_len - 1
            y = series_data[end_input : end_target]
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

        # 4. Return as tensors
        # Adding a dimension for features: (Seq_Len, Features)
        # If univariate, Features = 1.
        return {
            'x': torch.tensor(x).unsqueeze(-1),
            'y': torch.tensor(y).unsqueeze(-1),
            # 'mask': None  # Placeholder for mask if needed
        }

