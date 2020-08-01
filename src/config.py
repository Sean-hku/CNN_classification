# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import torch
from src.opt import opt

# Training
datasets = {"ceiling": ["butterfly", "frog"]}
if opt.backbone == 'inception':
    input_size = 299
else:
    input_size = 224

device = "cuda:0"


