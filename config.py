# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import torch

'''
基本参数
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]

input_size_dict = {"inception":299, "resnet18":224, "resnet34":224, "resnet50":224, "resnet101":224, "resnet152":224,
                   "squeezenet":224, "LeNet": 28, "mobilenet":224, "shufflenet": 224}

batch_size_dict = {"inception":32, "resnet18":64, "resnet34":64, "resnet50":64, "resnet101":32, "resnet152":32,
                   "squeezenet":128, "LeNet": 128, "mobilenet":64, "shufflenet": 128}

epochs_dict = {"inception":20, "resnet18":20, "resnet34":20, "resnet50":20, "resnet101":20, "resnet152":20,
                   "squeezenet":20, "LeNet": 20, "mobilenet":20, "shufflenet": 20}


'''
模型训练参数
'''
train_type = 'golf_ske'
pre_train_model_name = "resnet18"


config_data_path = os.path.join("data", train_type)
model_save_path = os.path.join("models/saved/", train_type)
feature_extract = False


input_size = input_size_dict[pre_train_model_name]
epoch = epochs_dict[pre_train_model_name]
batch_size = batch_size_dict[pre_train_model_name]


golf_ske_label_dict = {"backswing": 0, "standing": 1, "finish": 2}
if train_type == "golf_ske":
    img_label_dict = golf_ske_label_dict
else:
    raise ValueError("Your type is wrong. Please check again")

train_class_nums = len(img_label_dict)

'''
# 模型测试参数
'''
test_type = "golf"

golf_label = ["Backswing", "Standing", "FollowThrough"]
test_model_path = "test/model/golf_ske_shufflenet_2019-10-11-12-42-10.pth"
test_sample_path = 'test/test_golf'

if test_type == "golf":
    test_label = golf_label
else:
    raise ValueError("Your type is wrong. Please check again")


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

