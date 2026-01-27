from tsl.metrics import torch_metrics


class MetricsLogger:
    def __init__(self):
        self.log_metrics = {'mae': torch_metrics.MaskedMAE(),
                            "mae_at_3_days": torch_metrics.MaskedMAE(at=2),
                            "mae_at_6_days": torch_metrics.MaskedMAE(at=5),
                            "mae_at_12_days": torch_metrics.MaskedMAE(at=11),
                            "mae_at_14_days": torch_metrics.MaskedMAE(at=13),
                            'mre': torch_metrics.MaskedMRE(),
                            'mse': torch_metrics.MaskedMSE()}
        
    def filter_metrics(self, metrics: list) -> dict:
        """
        Filter the metrics to only include those that are in the list metrics.
        """

        filtered_metrics = {}
        for metric in metrics:
            if metric in self.log_metrics:
                filtered_metrics[metric] = self.log_metrics[metric]
            else:
                print(f"Metric {metric} is not defined in log_metrics.")
                
        return filtered_metrics
    

        
