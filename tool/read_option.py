import torch

option_path = "weight/test/18/option.pth"
try:
    info = torch.load(option_path)
except:
    info = torch.load("../" + option_path)
print(info)
