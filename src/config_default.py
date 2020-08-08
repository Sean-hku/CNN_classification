# -*- coding:utf-8 -*-
from __future__ import print_function
from src.opt import opt
device = "cuda:0"
computer = "laptop"

# Training
datasets = {"ceiling": ["butterfly", "frog"]}

if opt.backbone == 'inception':
    input_size = 299
else:
    input_size = 224

freeze_pretrain = {"mobilenet": [155, "classifier"],
                   "shufflenet": [167, "fc"],
                   "mnasnet": [155, "classifier"],
                   "resnet18": [59, "fc"],
                   "squeezenet": [49, "classifier"],
                   "resnet34": [107, "fc"],
                   }

warm_up = {0: 0.1, 1: 0.5}
bad_epochs = {30: 0.1}
patience_decay = {1: 0.5, 2: 0.5, 3: 0}


# Testing
test_model_path = "exp/pre_train_model/mnasnet.pth"
test_img = "tmp/cat.jpeg"
