import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data_df, 
                 input_len, 
                 pred_len, 
                 contain_missing_values=False,
                 cols_to_use='series_value',
                 training_mode="normal"):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_window = input_len + pred_len
        self.training_mode = training_mode
        
        # Pre-convert all series to tensors once during init
        self.all_series = []
        self.samples_map = []
        
        for series_idx, (_, row) in enumerate(data_df.iterrows()):
            # Convert to tensor immediately - single allocation
            series_tensor = torch.from_numpy(
                np.asarray(row[cols_to_use], dtype=np.float32)
            ).unsqueeze(-1)  # (T, 1) - add feature dim once
            
            series_length = len(series_tensor)
            self.all_series.append(series_tensor)
            
            max_start_idx = series_length - self.total_window + 1
            if max_start_idx > 0:
                # Batch append is faster than loop
                self.samples_map.extend(
                    (series_idx, t) for t in range(max_start_idx)
                )
        
        # Convert to numpy for faster indexing
        self.samples_map = np.array(self.samples_map, dtype=np.int32)
        print(f"Dataset created. Total samples: {len(self.samples_map)}")

        # Shuffle samples initially for randomness (data might be not very stationary)
        self.shuffle_samples()

    def __len__(self):
        return len(self.samples_map)
    
    def shuffle_samples(self):
        np.random.shuffle(self.samples_map)

    def __getitem__(self, idx):
        series_idx, start_t = self.samples_map[idx]
        series = self.all_series[series_idx]
        
        end_input = start_t + self.input_len
        end_target = end_input + self.pred_len
        
        if self.training_mode == "normal":
            x = series[start_t:end_input]
            y = series[end_input:end_target]
        elif self.training_mode == "teacher_forcing":
            x = series[start_t:end_target - 1]
            y = series[end_input:end_target]
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
        
        # Slicing tensors returns views - no copy, very fast
        return {'x': x, 'y': y}