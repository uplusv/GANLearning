from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseModel(ABC):
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda' if opts['num_gpu'] != 0 else 'cpu')
        self.model = []
        self.ema_model = []
        self.schedulers = []
        self.optimizers = []
    
    @abstractmethod
    def feed_data(self, data):
        """ 
            tranform raw data to the form models want
        """
        pass

    @abstractmethod
    def optimize_params(self):
        pass

    def setup_optimizer(self, optimizer_type, params, lr, **kwargs):
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optimizer_type == 'ranger':
            from training.ranger import Ranger
            optimizer = Ranger(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optimizer_type} is not supperted yet.')
        return optimizer

    def setup_schedulers(self):
        pass

    def update_learning_rate(self, iter):
        pass

    def model_ema(self, decay=0.999):
        raw_dict = dict(self.model.named_parameters())
        ema_dict = dict(self.ema_model.named_parameters())

        for k in ema_dict.keys():
            ema_dict[k].data.mul_(decay).add_(raw_dict[k].data, alpha=1 - decay)

    def save_checkpoints(self):
        pass

    def resume(self):
        pass