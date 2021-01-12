model_name = 'resnet18'
classes = ['drown', 'stand']
num_classes = len(classes)
thresh = 60
model_path = '/media/hkuit164/WD20EJRX/CNN_classification/weight/test/default/default_resnet18_2cls_24.pth'
pruned_cfg_file = './cfg.txt'
pruned_model_file = './new_model.pth'

'''----------------------------------------------------------------------------'''

test_model_path = './new_model.pth'
img_path = '/media/hkuit164/WD20EJRX/5_pics/Drown/0044_80.jpg'
