from pytorch_lightning.callbacks import Callback

class MetricsHistory(Callback):
    def __init__(self, log_metrics, log_dir=None):
        super().__init__()

        self.log_dir = log_dir
        self.log_metrics = log_metrics

        # Initialize lists to store metrics
        for metric in log_metrics:
            setattr(self, f'train_{metric}', [0])
            setattr(self, f'val_{metric}', [])
            setattr(self, f'test_{metric}', [])

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        for metric in self.log_metrics:
            if f'train_{metric}' in m:
                getattr(self, f'train_{metric}').append(m[f'train_{metric}'].cpu().item())

    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        for metric in self.log_metrics:
            if f'val_{metric}' in m:
                getattr(self, f'val_{metric}').append(m[f'val_{metric}'].cpu().item())

    def on_test_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        for metric in self.log_metrics:
            if f'test_{metric}' in m:
                getattr(self, f'test_{metric}').append(m[f'test_{metric}'].cpu().item())

class Wandb_callback(Callback):
    def __init__(self, log_metrics, log_dir=None, run=None):
        super().__init__()
        self.log_dir = log_dir
        self.run = run
        self.epoch = 0
        self.log_metrics = log_metrics

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        log_dict = {"train_"+k: m["train_"+k].cpu().item() for k in self.log_metrics}

        self.run.log(log_dict, step = self.epoch)
    
    def on_validation_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        log_dict = {"val_"+k: m["val_"+k].cpu().item() for k in self.log_metrics}

        self.run.log(log_dict, step = self.epoch)

        self.epoch += 1

    def on_test_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        log_dict = {"test_"+k: m["test_"+k].cpu().item() for k in self.log_metrics}

        self.run.log(log_dict, step = self.epoch)
