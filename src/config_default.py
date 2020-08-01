# -*- coding:utf-8 -*-
from __future__ import print_function
from src.opt import opt
device = "cuda:0"

# Training
datasets = {"ceiling": ["butterfly", "frog"]}
if opt.backbone == 'inception':
    input_size = 299
else:
    input_size = 224


# Testing
test_model_path = "exp/pre_train_model/mnasnet.pth"
test_img = "tmp/cat.jpeg"
