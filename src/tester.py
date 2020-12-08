# -*- coding:utf-8 -*-
from __future__ import print_function
from src.config import device, input_size
from src.model import CNNModel, LeNet
import torch
import numpy as np
from torch import nn
from src.utils import image_normalize
import os


class ModelInference(object):
    def __init__(self, class_nums, pre_train_name, model_path):
        if "LeNet" not in pre_train_name:
            self.CNN_model = CNNModel(class_nums, pre_train_name).model.to(device)
        else:
            self.CNN_model = LeNet(class_nums)
        self.CNN_model.load_state_dict(torch.load(model_path, map_location=device))
        if device != "cpu":
            self.CNN_model.cuda()

    def predict(self, img):
        img_tensor_list = []
        img_tensor = image_normalize(img, size=input_size)
        img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
        if len(img_tensor_list) > 0:
            input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
            res_array = self.__predict_image(input_tensor)
            return res_array

    def __predict_image(self, image_batch_tensor):
        self.CNN_model.eval()
        self.image_batch_tensor = image_batch_tensor.cuda()
        outputs = self.CNN_model(self.image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        print(outputs)
        return np.asarray(outputs_tensor)

    def to_onnx(self, name="model.onnx"):
        torch_out = torch.onnx.export(self.CNN_model, self.image_batch_tensor, name, verbose=False,)
#                                      input_names=in_names, output_names=out_names)

    def to_libtorch(self):
        example = torch.rand(2, 3, 224, 224).cuda()
        self.CNN_model.eval()
        traced_model = torch.jit.trace(self.CNN_model, example)
        traced_model.save("drown_res.pt")
