from tsl.engines import Predictor
from typing import Optional, Callable, Mapping, Type, Union, List, Dict
from torchmetrics import Metric
from torch import nn, Tensor
from torch.nn import functional as F
import einops
import os 
import torch

class WrapPredictor(Predictor):
    """
    A class that extends the TSL Predictor to implement a custom prediction method.
    This class can be used to predict future values based on past observations.
    """

    def __init__(self,
                 model: Optional[nn.Module] = None,
                 loss_fn: Optional[Callable] = None,
                 scale_target: bool = False,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 warm_up: int = 5,
                 n_nodes: int = 1,
                 transform: Optional[Mapping[str, Callable]] = None,
                 *,
                 model_class: Optional[Type] = None,
                 model_kwargs: Optional[Mapping] = None,
                 optim_class: Optional[Type] = None,
                 optim_kwargs: Optional[Mapping] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None,
                 sampling: int = -1,
                 batch_size: int = 32):
        
        super(WrapPredictor, self).__init__(       model=model,
                                                    model_class=model_class,
                                                    model_kwargs=model_kwargs,
                                                    optim_class=optim_class,
                                                    optim_kwargs=optim_kwargs,
                                                    loss_fn=loss_fn,
                                                    scale_target=scale_target,
                                                    metrics=metrics,
                                                    scheduler_class=scheduler_class,
                                                    scheduler_kwargs=scheduler_kwargs)
        self.sampling = sampling
        self.transform = transform
        self.batch_size = batch_size

    def predict_batch(self, 
                        batch,
                        preprocess: bool = False, 
                        postprocess: bool = True,
                        return_target: bool = False,
                        maybe_x: Optional[Tensor] = None,
                        maybe_y: Optional[Tensor] = None,
                        **forward_kwargs):

            inputs = batch.get('x')
            targets = batch.get('y')
            # mask = batch.get('mask')

            if maybe_y is not None:
                targets = maybe_y

            if preprocess:
                transform = self.transform.get('target')
                if transform is not None:
                    inputs = transform.transform(inputs)

            if forward_kwargs is None:
                forward_kwargs = dict()

            if maybe_x is not None:
                x = maybe_x
                y_hat = self.forward(x, **inputs, **forward_kwargs)
            else:
                y_hat = self.forward(inputs, **forward_kwargs)

            # IMMA SKIP THIS FOR NOW
            # Rescale outputs
            # if postprocess:
            #     trans = self.transform.get('target')
            #     if trans is not None:
            #         y_hat = trans.inverse_transform(y_hat)

            if return_target:
                y = targets.get('y')
                return y, y_hat
            
            return y_hat
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        
        # Unpack batch
        x, y, mask, transform = self._unpack_batch(batch)

        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        output = dict(**y, y_hat=y_hat)
        if mask is not None:
            output['mask'] = mask
        return output

    def training_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:

        mask = batch.get('mask')

        # if we sample, preserve only the input and target of the half with the highest std
        if self.sampling != -1:
            if 'x' in batch:
                x = batch['x']
                y = batch['y']
                std = y.std(dim=(1,2,3)) + x.std(dim=(1,2,3))

                if self.sampling == 0:
                    idx = std.nonzero(as_tuple=False).squeeze(1)
                else:
                    idx = std.topk(x.shape[0] // self.sampling).indices

                # drop all zero std samples
                batch['x'] = x[idx, :]
                batch['y'] = y[idx, :]
                
                if 'u' in batch:
                    batch['u'] = batch['u'][idx, :]

                if 'enable_mask' in batch.keys():
                    batch['enable_mask'] = batch['enable_mask'][idx, :]

                mask = mask[idx, :]

            else:
                raise ValueError("Sampling is only supported for batches with 'x'.")

        y = y_loss = batch['y']

        # Compute predictions and compute loss
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.train_metrics.update(y_hat, y, mask)
        self.log_metrics(self.train_metrics, batch_size=self.batch_size)
        self.log_loss('train', loss, batch_size=self.batch_size)

        return loss
    
    def validation_step(self, batch: Union[Tensor, Dict[str, Tensor]], batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:
        """"""
        y = y_loss = batch['y']
        mask = batch.get('mask')

        # Compute predictions
        y_hat_loss = self.predict_batch(batch, preprocess=False,
                                             postprocess=not self.scale_target)
        y_hat = y_hat_loss.detach()

        # Scale target and output, eventually
        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        # Compute loss
        val_loss = self.loss_fn(y_hat_loss, y_loss, mask)

        # Logging
        self.val_metrics.update(y_hat, y, mask)
        self.log_metrics(self.val_metrics, batch_size=self.batch_size)
        self.log_loss('val', val_loss, batch_size=self.batch_size)
        return val_loss
    
    def test_step(self, batch: Union[Tensor, Dict[str, Tensor]], _batch_idx: int = 0) -> Union[Tensor, Dict[str, Tensor]]:

        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False,
                                        postprocess=True)

        y = batch['y']
        mask = batch.get('mask')
        test_loss = self.loss_fn(y_hat, y, mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_metrics, batch_size=self.batch_size)
        self.log_loss('test', test_loss, batch_size=self.batch_size)

        return test_loss

    
    def save_model(self, filepath):
            """
            Saves the model weights and necessary metadata to a file.
            """
            # Ensure the directory exists
            dir_name = os.path.dirname(filepath)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # Create a checkpoint dictionary
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_kwargs': self.model_kwargs,
                'state_dict': self.state_dict(),  # Includes optimizer states if needed
            }
            
            torch.save(checkpoint, filepath)
            print(f"Model saved successfully to {filepath}")

    def load_model(self, filepath):
            """
            Loads the model weights and necessary metadata from a file.
            """
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No such file: {filepath}")

            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))

            # Load model state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load predictor state dict (includes optimizer states if needed)
            self.load_state_dict(checkpoint['state_dict'])

            print(f"Model loaded successfully from {filepath}")
