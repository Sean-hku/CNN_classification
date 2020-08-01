from __future__ import print_function
import torch
import src.config as config
from torch.utils.data import Dataset
from utils import image_normalize
from tools.adjust_val import ImgAdjuster
import os
from src.opt import opt


class MyDataset(Dataset):
    def __init__(self, img_path, image_label_dict, image_processor=image_normalize, size=224):
        img_dir_name, img_dir_label, self.img_name, self.img_label = [], [], [], []
        self.size = size

        for idx, cls in enumerate(image_label_dict):
            img_dir_name.append(os.path.join(img_path, cls))
            img_dir_label.append(idx)

        for dir_name, dir_label in zip(img_dir_name, img_dir_label):
            img_file_names = os.listdir(os.path.join(dir_name))
            for img_name in img_file_names:
                self.img_name.append(os.path.join(dir_name, img_name))
                self.img_label.append(dir_label)

        self.image_processor = image_processor

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        image_name = self.img_name[item]
        label = self.img_label[item]
        try:
            image_object = self.image_processor(image_name, size=self.size)
            return image_name, image_object, label
        except Exception as e:
            print("error read ", image_name, e)
            # os.remove(image_name)
            image_name = self.img_name[0]
            _label = self.img_label[0]
            _image_object = self.image_processor(image_name, size=self.size)
            return image_name, _image_object, _label


class DataLoader(object):
    def __init__(self, data_dir, batch_size, inp_size=224):
        adjust = ImgAdjuster(opt.val_ratio, data_dir)
        adjust.run()
        self.image_datasets = {x: MyDataset(os.path.join(data_dir, x), config.datasets[opt.dataset], size=inp_size)
                               for x in ['train', 'val']}
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size,
                                                                shuffle=True, num_workers=opt.num_worker)
                            for x in ['train', 'val', ]}


# class DataLoader_Auto(object):
#     def __init__(self, data_src, label_dict, batch_size_auto, inp_size):
#         adjust = ImgAdjuster(opt.val_ratio, data_src)
#         adjust.run()
#         self.image_datasets = {x: MyDataset(os.path.join(data_src, x), label_dict, size=inp_size)
#                                for x in ['train', 'val']}
#         self.dataloaders_dict = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=batch_size_auto,
#                                                                 shuffle=True, num_workers=1)
#                                  for x in ['train', 'val',]}