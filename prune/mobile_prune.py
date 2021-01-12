from src.tester import ModelInference
from prune.utils import *


model_path = '/media/hkuit164/WD20EJRX/CNN_classification/weight/test/default/default_mobilenet_2cls_best.pth'
name = "mobilenet"
classes = ['drown','stand']
num_classes = len(classes)
thresh = 60
model = ModelInference(num_classes, name, model_path,cfg = []).CNN_model
print(model)