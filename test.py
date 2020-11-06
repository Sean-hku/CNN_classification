from src import config
from src.dataloader import DataLoader
from src.tester import ModelInference
import torch
import matplotlib.pyplot as plt
import  numpy as np
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
def get_pretrain(model_path):
    if "_resnet18" in model_path:
        name = "resnet18"
    elif "_resnet50" in model_path:
        name = "resnet50"
    elif "_resnet34" in model_path:
        name = "resnet34"
    elif "_resnet101" in model_path:
        name = "resnet101"
    elif "_resnet152" in model_path:
        name = "resnet152"
    elif "_inception" in model_path:
        name = "inception"
    elif "_mobilenet" in model_path:
        name = "mobilenet"
    elif "_shufflenet" in model_path:
        name = "shufflenet"
    elif "_LeNet" in model_path:
        name = "LeNet"
    elif "_squeezenet" in model_path:
        name = "squeezenet"
    elif "mnasnet" in model_path:
        name = "mnasnet"
    elif "LeNet" in model_path:
        name = "LeNet"
    else:
        raise ValueError("Wrong name of pre-train model")
    return name
def test():
    # with torch.no_grad():
    drown_ls = torch.FloatTensor([]).to("cuda:0")
    stand_ls = torch.FloatTensor([]).to("cuda:0")
    labels_ls_drown = torch.LongTensor([]).to("cuda:0")
    labels_ls_stand = torch.LongTensor([]).to("cuda:0")
    model_path = config.test_model_path
    img_path = config.test_img
    batch_size = 16
    data_loader = DataLoader(img_path, batch_size)
    classes = ['drown','stand']
    num_classes = len(classes)
    pbar = tqdm(enumerate(data_loader.dataloaders_dict['val']), total=len(data_loader.dataloaders_dict['val']))
    for i,(names, inputs, labels) in pbar:
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")
        pre_name = get_pretrain(model_path)
        model = ModelInference(num_classes, pre_name, model_path)
        outputs = model.sport_model(inputs)
        _, preds = torch.max(outputs, 1)
        # print(torch.sigmoid(outputs))
        drown = torch.index_select(torch.sigmoid(outputs),1,torch.tensor([1]).to("cuda:0"))
        drown = drown.view(1,-1).squeeze()
        stand = torch.index_select(torch.sigmoid(outputs), 1, torch.tensor([0]).to("cuda:0"))
        stand = stand.view(1, -1).squeeze()
        # print(drown)
        # print('~~~~~~'+str(labels))
        drown_ls = torch.cat((drown_ls,drown),0)
        stand_ls = torch.cat((stand_ls,stand),0)
        labels_ls_drown = torch.cat((labels_ls_drown, labels), 0)
        labels_ls_stand = torch.cat((labels_ls_stand, torch.add(1,-labels).long()), 0)
    precision, recall, thresholds = precision_recall_curve(labels_ls_drown.cpu().detach().numpy(), np.array(drown_ls.squeeze().cpu().detach().numpy()))
    # labels_ls_stand = labels_ls_stand.ge(-0.1)
    # precision, recall, thresholds = precision_recall_curve(labels_ls_stand.cpu().detach().numpy(),
    #                                                np.array(stand_ls.squeeze().cpu().detach().numpy()))
    plt.figure(1)
    plt.plot(precision, recall)
    plt.show()
if __name__ == '__main__':
    test()