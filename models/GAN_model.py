from atexit import register
import torch
from basic_model import BaseModel
from DCNGAN_arch import DCNGAN
from utils.register import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class GAN(BaseModel):
    def __init__(self, opts):
        super().__init__(opts)
        self.model = DCNGAN()

    def feed_data(self, data):
        return super().feed_data(data)
    
    def optimize_params(self):
        return super().optimize_params()
