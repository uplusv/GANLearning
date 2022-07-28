from distutils.command.build import build
import os
import torch
from models import build_model
from data import build_dataset

class Coach:
    def __init__(self, opts) -> None:
        self.opts = opts
        self.model = build_model(opts)
        self.dataset = self._config_datasets()
        self.metrics = {}
        self.global_step = 0

        if opts["resume"]:
            self.model.resume()
        
        if opts["use_wandb"]:
            from utils.logger import WandbLogger
            self.logger = WandbLogger(self.opts)
        else:
            from utils.logger import LocalLogger
            self.logger = LocalLogger()

    def _config_datasets(self):
        datasets = {}
        datasets['train'] = build_dataset(self.opts, type="train")
        datasets['valid'] = build_dataset(self.opts, type="valid")
        datasets['test'] = build_dataset(self.opts, type="test")
        return datasets

    def train(self):
        for epoch in self.opts["train"]["epoch"]:

            for i, train_data in self.dataset["train"]:
                self.model.feed_data(train_data)
                loss_dict, self.global_step = self.model.optimize_params(self.global_step) # potential gradient accumulate
                self.model.update_learning_rate(self.global_step) 

            if self.global_step % self.opts["train"]["log_freq"]:
                self.logger.log(loss_dict)
            if self.global_step % self.opts["val"]["val_freq"]:
                self.valid()

    def valid(self):
        for i, valid_data in self.dataset["valid"]:
            self.model.feed_data(valid_data)
            log_dict = self.model.validate()
        
        for name, metric in self.metrics.items():
            log_dict[name] = metric()
        
        self.logger.log(prefix="valid", metrics_dict=log_dict, global_step=self.global_step)

