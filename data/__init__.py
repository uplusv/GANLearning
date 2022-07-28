import importlib
from copy import deepcopy
import os
from os import path as osp

from utils.register import DATASET_REGISTRY

__all__ = ['build_dataset']

# automatically scan and import dataset modules for registry
# scan all the files under the 'data' folder and collect files ending with
# '_dataset.py'
dataset_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in os.listdir(dataset_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]


def build_dataset(opt, type="train"):
    """Build dataset from options.

    Args:
        opt (dict): Configuration. It must contain:
            dataset:
                train/valid/test:
                    dataset_type (str): dataset type.
    """
    opt = deepcopy(opt)
    dataset = DATASET_REGISTRY.get(opt['dataset'][type]['dataset_type'])(opt)
    # TODO: add logger
    # logger = get_root_logger()
    # logger.info(f'dataset [{dataset.__class__.__name__}] is created.')
    return dataset
