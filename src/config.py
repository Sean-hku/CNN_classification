# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import torch
from src.opt import opt

'''
基本参数
'''

datasets = {"ceiling": ["butterfly", "frog"]}
if opt.backbone == 'inception':
    input_size = 299
else:
    input_size = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




'''
模型训练参数
'''
train_type = 'underwater'
pre_train_model_name = "mobilenet"


config_data_path = os.path.join("data", train_type)
model_save_path = os.path.join("weight/saved/", train_type)


input_size_dict = {"inception":299, "resnet18":224, "resnet34":224, "resnet50":224, "resnet101":224, "resnet152":224,
                   "squeezenet":224, "LeNet": 28, "mobilenet":224, "shufflenet": 224}

batch_size_dict = {"inception":32, "resnet18":64, "resnet34":64, "resnet50":64, "resnet101":32, "resnet152":32,
                   "squeezenet":128, "LeNet": 128, "mobilenet":64, "shufflenet": 128}

epochs_dict = {"inception":20, "resnet18":10, "resnet34":20, "resnet50":20, "resnet101":20, "resnet152":20,
                   "squeezenet":20, "LeNet": 20, "mobilenet":20, "shufflenet": 20}

input_size = input_size_dict[pre_train_model_name]
epoch = epochs_dict[pre_train_model_name]
batch_size = batch_size_dict[pre_train_model_name]


golf_ske_label_dict = {"backswing": 0, "standing": 1, "finish": 2}
# ceiling_dict = {"BA":0, "BR":1, "BU":2, "FR":3, "SIDE BR":4, "SIDE FR":5, "STAND":6}
# ceiling_dict = {"back":0, "butterfly":1, "freestyle":2, "frog":3, "side_freestyle":4, "side_frog":5, "standing":6}
ceiling_dict = {"freestyle":0, "frog":1, "side_freestyle":2, "side_frog":3}
underwater_dict = {"drown":0, "floating": 1, "stand walk":2}

if train_type == "golf_ske":
    img_label_dict = golf_ske_label_dict
elif train_type == "ceiling":
    img_label_dict = ceiling_dict
elif train_type == "underwater":
    img_label_dict = underwater_dict
else:
    raise ValueError("Your type is wrong. Please check again")

train_class_nums = len(img_label_dict)


# 自动训练参数
auto_train_type = 'golf_ske'
auto_train_folder = "test"

val_ratio_ls = [0.1]
epoch_ls = [1,2]
pre_train_ls = ["mobilenet", "shufflenet"]
learning_rate_ls = [0.001]

auto_golf_ske_label_dict = {"backswing": 0, "finish": 1, "standing": 2}

if auto_train_type == "golf_ske":
    auto_train_label_dict = auto_golf_ske_label_dict
else:
    raise ValueError("Wrong train type!")

auto_train_class_num = len(auto_train_label_dict)


# 自动测试参数
model_folder = 'test/model/1014'
log = os.path.join(model_folder, "log.txt")
autotest_threshold = 0.4
positive_sample = r''
negative_sample = r''

