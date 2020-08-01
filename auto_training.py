from src.model import CNNModel, LeNet
from src.trainer import train_model
from src.dataloader import DataLoader_Auto
import src.config as config
import torch.nn as nn
import torch.optim as optim
import time
import os
from tools import adjust_val

data_type = config.auto_train_type
folder_name = config.auto_train_folder

val_ratio_ls = config.val_ratio_ls
epoch_ls = config.epoch_ls
pre_train_ls = config.pre_train_ls
label_dict = config.auto_train_label_dict

class_num = len(label_dict)
feature_extract = config.feature_extract
device = config.device
learning_rate_ls = config.learning_rate_ls

model_save_folder = os.path.join("models/auto_train_saved/", folder_name, "models")
log_save_folder = os.path.join("models/auto_train_saved/", folder_name, "log")


class AutoTrainer(object):
    def __init__(self, pre_name, epo, lr):
        self.pre_train_model_name = pre_name
        self.data_src = os.path.join("data", data_type)
        self.epoch = epo

        self.model_str = data_type + "_%s_%s.pth" % (self.pre_train_model_name, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        self.model_save_path = os.path.join(model_save_folder, self.model_str)
        self.log_save_path = os.path.join(log_save_folder, self.model_str.replace(".pth", "_log.txt"))
        self.record_path = os.path.join("models/auto_train_saved/", folder_name, "result.csv")

        self.sport_model = self.__load_model()
        params_to_update = self.sport_model.parameters()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.Adam(params_to_update, lr=lr)
        self.data_loader = DataLoader_Auto(self.data_src, label_dict, config.batch_size_dict[self.pre_train_model_name],
                                           config.input_size_dict[self.pre_train_model_name])

    def __load_model(self):
        if self.pre_train_model_name == "LeNet":
            model = LeNet(class_num).to(device)
        else:
            model = SportModel(class_num, self.pre_train_model_name, feature_extract).model.to(device)
        return model
    #
    # def __get_dataloader(self):
    #     return DataLoader_Auto(self.data_src, label_dict, config.batch_size_dict[self.pre_train_model_name],
    #                                        config.input_size_dict[self.pre_train_model_name])

    def record(self):
        with open(self.record_path, 'a') as f:
            out = ','.join([self.model_str, self.pre_train_model_name, str(self.epoch), str(val), str(learning_rate)]) + '\n'
            f.write(out)

    def failed_record(self):
        with open(self.record_path, 'a') as file:
            file.write("Something wrong happens when training {}\n".format(self.model_str))

    def auto_train(self):
        train_model(self.sport_model, self.data_loader.dataloaders_dict, self.criterion, self.optimizer_ft, self.epoch,
                    self.pre_train_model_name == "inception", self.model_save_path, self.log_save_path)
        print("train model done, save model to %s" % self.model_save_path)


if __name__ == "__main__":
    cnt = 0
    total_num = len(epoch_ls) * len(pre_train_ls) * len(val_ratio_ls) * len(learning_rate_ls)
    os.makedirs(model_save_folder, exist_ok=True)
    os.makedirs(log_save_folder, exist_ok=True)

    with open(os.path.join("models/auto_train_saved/", folder_name, "result.csv"), "w") as f:
        f.write("model_name,pretrain_model,epoch,val ratio,learning-rate\n")

    for val in val_ratio_ls:
        for cls in label_dict.keys():
            IA = adjust_val.ImgAdjuster(val, data_type, cls)
            IA.run()
        for pre_model in pre_train_ls:
            for epoch in epoch_ls:
                for learning_rate in learning_rate_ls:
                    cnt += 1
                    print("\n\nBeginning to train: {}/{}".format(cnt, total_num))
                    print("The validation ratio is {}".format(val))
                    print("It will train {} epochs".format(epoch))
                    print("The pre_train model is {}".format(pre_model))
                    AutoTrain = AutoTrainer(pre_model, epoch, learning_rate)
                    # try:
                    AutoTrain.auto_train()
                    AutoTrain.record()
                    # except:
                    #     AutoTrain.failed_record()
