import torch

option_path = "/media/hkuit164/WD20EJRX/CNN_classification/weight/underwater_action-2_class/1/option.pth"
try:
    info = torch.load(option_path)
except:
    info = torch.load("../" + option_path)
print(info)
