import torch
from basic_model import BaseModel

class GAN(BaseModel):
    def __init__(self, opts):
        super().__init__(opts)

    def feed_data(self, data):
        return super().feed_data(data)
    
    def optimize_params(self):
        return super().optimize_params()
