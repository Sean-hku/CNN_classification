# -*- coding:utf-8 -*-
from __future__ import print_function
from src.model import CNNModel, LeNet
from src.trainer import train_model
from src.dataloader import DataLoader
import src.config as config
import torch.nn as nn
import torch.optim as optim
import os
from src.opt import opt
import torch
import sys
from src import utils


device = config.device
feature_extract = opt.freeze
num_epochs = opt.epoch
data_name = opt.dataset

class_nums = len(config.datasets[data_name])
data_dir = os.path.join("data", data_name)
backbone = opt.backbone
batch_size = opt.batch

modelID = opt.expID
model_save_path = os.path.join("weight/{}/{}".format(opt.expFolder, modelID))
os.makedirs(model_save_path, exist_ok=True)

if __name__ == "__main__":
    print(opt)
    cmd_ls = sys.argv[1:]
    cmd = utils.generate_cmd(cmd_ls)

    if backbone == "LeNet":
        model = LeNet(class_nums).to(device)
    else:
        model = CNNModel(class_nums, backbone, feature_extract).model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")

    if opt.loadModel:
        model_path = os.path.join("weight/pre_train_model/%s.pth" % backbone)
        model.load_state_dict(torch.load(model_path, map_location=device))

    if feature_extract > 0:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    if opt.optMethod == "adam":
        optimizer_ft = optim.Adam(params_to_update, lr=opt.LR)
    else:
        raise ValueError("Wrong optimizer name")

    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(data_dir, batch_size)

    is_inception = backbone == "inception"
    train_model(model, data_loader.dataloaders_dict, criterion, optimizer_ft, cmd, is_inception=is_inception,
                model_save_path=model_save_path)

