from src.tester import ModelInference
import cv2
import torch
import numpy as np
import torch.nn as nn




def image_normalize(img_name, size=224):
    image_normalize_mean = [0.485, 0.456, 0.406]
    image_normalize_std = [0.229, 0.224, 0.225]
    if isinstance(img_name, str):
        image_array = cv2.imread(img_name)
    else:
        image_array = img_name
    image_array = cv2.resize(image_array, (size, size))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
    image_array = image_array.transpose((2, 0, 1))
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor

def predict( CNN_model,img):
    img_tensor_list = []
    img_tensor = image_normalize(img)
    img_tensor_list.append(torch.unsqueeze(img_tensor, 0))
    if len(img_tensor_list) > 0:
        input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
        res_array = predict_image(CNN_model,input_tensor)
        return res_array

def predict_image(CNN_model, image_batch_tensor):
    CNN_model.eval()
    image_batch_tensor = image_batch_tensor.cuda()
    outputs = CNN_model(image_batch_tensor)
    outputs_tensor = outputs.data
    m_softmax = nn.Softmax(dim=1)
    outputs_tensor = m_softmax(outputs_tensor).to("cpu")
    print(outputs)
    return np.asarray(outputs_tensor)

classes = ['drown','stand']
num_classes = len(classes)
name = 'resnet18'
model_path = '/media/hkuit164/WD20EJRX/CNN_classification/prune/new_model.pth'
model = ModelInference(num_classes, name, model_path,cfg ='prune').CNN_model
img = cv2.imread('/media/hkuit164/WD20EJRX/5_pics/Drown/0044_80.jpg')
res = predict(model,img)
print(res)