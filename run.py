import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from model.model_builder import init_model
from model import *
from init_config import *
from easydict import EasyDict as edict
import sys
from trainer.ccm_trainer import Trainer
import copy
import numpy as np 
import random


def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    config, writer = init_config("config/final_config.yml", sys.argv)

    config.num_classes = 19

    model = init_model(config)


    trainer = Trainer(model, config, writer)

    trainer.train()


if __name__ == "__main__":
    main()
