import torch
import torchvision.models as models

from src.tester import ModelInference
# model = ModelInference(class_nums=2,pre_train_name='resnet34',model_path='/media/hkuit164/WD20EJRX/CNN_classification/weight/underwater_action-2_class_resnet34/7/7_resnet34_9_decay2.pth',cfg = [])
model =models.resnet34()
model.load_state_dict(torch.load('/media/hkuit164/WD20EJRX/CNN_classification/weight/underwater_action-2_class_resnet34/7/7_resnet34_9_decay2.pth'))
model.eval()

x=torch.rand(1,3,416,416)
ts = torch.jit.trace(model,x)
ts.save('11.pt')
