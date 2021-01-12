import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
from src import config
from src.dataloader import DataLoader
from src.opt import opt
from src.tester import ModelInference
from src.utils import get_pretrain


def test(model_path, img_path, batch_size, num_classes):
    drown_ls = torch.FloatTensor([]).to("cuda:0")
    stand_ls = torch.FloatTensor([]).to("cuda:0")
    labels_ls_drown = torch.LongTensor([]).to("cuda:0")
    labels_ls_stand = torch.LongTensor([]).to("cuda:0")
    data_loader = DataLoader(img_path, batch_size)
    pbar = tqdm(enumerate(data_loader.dataloaders_dict['val']), total=len(data_loader.dataloaders_dict['val']))
    for i, (names, inputs, labels) in pbar:
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")
        pre_name = get_pretrain(model_path)
        model = ModelInference(num_classes, pre_name, model_path, cfg='default')
        outputs = model.CNN_model(inputs)
        _, preds = torch.max(outputs, 1)
        drown = torch.index_select(torch.sigmoid(outputs), 1, torch.tensor([1]).to("cuda:0"))
        drown = drown.view(1, -1).squeeze()
        stand = torch.index_select(torch.sigmoid(outputs), 1, torch.tensor([0]).to("cuda:0"))
        stand = stand.view(1, -1).squeeze()
        drown_ls = torch.cat((drown_ls, drown), 0)
        stand_ls = torch.cat((stand_ls, stand), 0)
        labels_ls_drown = torch.cat((labels_ls_drown, labels), 0)
        labels_ls_stand = torch.cat((labels_ls_stand, torch.add(1, -labels).long()), 0)
    precision, recall, thresholds = precision_recall_curve(labels_ls_drown.cpu().detach().numpy(),
                                                           np.array(drown_ls.squeeze().cpu().detach().numpy()))
    '''#for stand class
    labels_ls_stand = labels_ls_stand.ge(-0.1)
    precision, recall, thresholds = precision_recall_curve(labels_ls_stand.cpu().detach().numpy(),
                                                   np.array(stand_ls.squeeze().cpu().detach().numpy()))
                                                   '''
    plt.figure(1)
    plt.plot(precision, recall)
    plt.show()


if __name__ == '__main__':
    model_path = config.test_model_path
    img_path = config.test_img
    batch_size = opt.batch
    classes = config.classes
    num_classes = len(classes)

    with torch.no_grad():
        test(model_path,img_path,batch_size,num_classes)
