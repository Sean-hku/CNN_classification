import torch
import os

class_str = 'freestyle,frog,side_freestyle,side_frog,butterfly,back,standing'


def add_opt(option_path):
    try:
        info = torch.load(option_path)
    except:
        option_path = "../" + option_path
        info = torch.load(option_path)
    print(info)
    info.classes = class_str
    print(info)
    torch.save(info, option_path)


if __name__ == '__main__':
    src = "../weight/ceiling_action-7_class"
    for item in os.listdir(src):
        opt_path = os.path.join(src, item, "option.pth")
        add_opt(opt_path)