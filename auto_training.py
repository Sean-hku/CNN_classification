from src.model import CNNModel, LeNet
from src.trainer import train_model
from src.dataloader import DataLoader
import src.config as config
import torch.nn as nn
import torch.optim as optim
import time
import os
from src.opt import opt

data_type = opt.dataset
folder_name = opt.expID
label_dict = config.datasets[data_type]
class_num = len(label_dict)

feature_extract = opt.freeze
device = config.device
exp_name = opt.expFolder
lr = opt.LR

save_folder = os.path.join("exp", exp_name, folder_name)
os.makedirs(save_folder, exist_ok=True)


class AutoTrainer(object):
    def __init__(self):
        self.backbone = opt.backbone

        self.epoch = opt.epoch
        model_str = data_type + "_%s_%s.pth" % (self.backbone, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        self.model_save_path = os.path.join(save_folder, model_str)
        self.log_save_path = os.path.join(save_folder, model_str.replace(".pth", "_log.txt"))

        self.model = self.__load_model()
        params_to_update = self.model.parameters()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.Adam(params_to_update, lr=lr)
        self.data_loader = DataLoader(os.path.join("data", data_type), opt.batch, config.input_size)

    def __load_model(self):
        if self.backbone == "LeNet":
            model = LeNet(class_num).to(device)
        else:
            model = CNNModel(class_num, self.backbone, feature_extract).model.to(device)
        return model

    def auto_train(self):
        train_model(self.model, self.data_loader.dataloaders_dict, self.criterion, self.optimizer_ft, self.epoch,
                    self.backbone == "inception", self.model_save_path, self.log_save_path)
        print("train model done, save model to %s" % self.model_save_path)


if __name__ == "__main__":
    AT = AutoTrainer()
    AT.auto_train()

#     cnt = 0
#     total_num = len(epoch_ls) * len(pre_train_ls) * len(val_ratio_ls) * len(learning_rate_ls)
#     os.makedirs(model_save_folder, exist_ok=True)
#     os.makedirs(log_save_folder, exist_ok=True)
#
#     with open(os.path.join("models/auto_train_saved/", folder_name, "result.csv"), "w") as f:
#         f.write("model_name,pretrain_model,epoch,val ratio,learning-rate\n")
#
#     for val in val_ratio_ls:
#         for cls in label_dict.keys():
#             IA = adjust_val.ImgAdjuster(val, data_type, cls)
#             IA.run()
#         for pre_model in pre_train_ls:
#             for epoch in epoch_ls:
#                 for learning_rate in learning_rate_ls:
#                     cnt += 1
#                     print("\n\nBeginning to train: {}/{}".format(cnt, total_num))
#                     print("The validation ratio is {}".format(val))
#                     print("It will train {} epochs".format(epoch))
#                     print("The pre_train model is {}".format(pre_model))
#                     AutoTrain = AutoTrainer(pre_model, epoch, learning_rate)
#                     # try:
#                     AutoTrain.auto_train()
#                     AutoTrain.record()
#                     # except:
#                     #     AutoTrain.failed_record()
