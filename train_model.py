# -*- coding:utf-8 -*-
from __future__ import print_function
from src.model import SportModel, LeNet
from src.trainer import train_model
from src.dataloader import DataLoader
import config
import torch.nn as nn
import torch.optim as optim
import time
import os

device = config.device
feature_extract = config.feature_extract
num_epochs = config.epoch

class_nums = config.train_class_nums
pre_train_model_name = config.pre_train_model_name
model_type = config.train_type
batch_size = config.batch_size_dict[pre_train_model_name]

model_save_path = config.model_save_path

if __name__ == "__main__":
    os.makedirs(model_save_path, exist_ok=True)
    if pre_train_model_name == "LeNet":
        model = LeNet(class_nums).to(device)
    else:
        model = SportModel(class_nums, pre_train_model_name, feature_extract).model.to(device)

    params_to_update = model.parameters()
    print("Params to learn:")

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    optimizer_ft = optim.Adam(params_to_update, lr=0.001)
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(batch_size)

    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_str = model_type + "_%s_%s.pth" % (pre_train_model_name, time_str)
    log_save_path = os.path.join(model_save_path, model_str.replace(".pth", "_log.txt"))

    is_inception = pre_train_model_name == "inception"
    silent_detect_model, hist = train_model(model, data_loader.dataloaders_dict, criterion, optimizer_ft,
                                            num_epochs=num_epochs, is_inception=is_inception, model_save_path=
                                            os.path.join(model_save_path, model_str), log_save_path=log_save_path)

    # save model
    print("train model done, save model to %s" % os.path.join(model_save_path, model_str))
