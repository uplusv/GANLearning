"""
This file runs the main training/val loop
"""
import argparse
import os
import sys
import yaml
import shutil

sys.path.append(".")
sys.path.append("..")

from training.coach import Coach
from utils.common import ordered_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    with open(args.opt, mode='r') as f:
        opts = yaml.load(f, Loader=ordered_yaml()[0])

    if not os.path.exists(opts["exp_dir"]):
        os.makedirs(opts["exp_dir"])
    shutil.copy(args.opt, opts["exp_dir"])

    coach = Coach(opts)
    coach.train()


if __name__ == '__main__':
    main()
