# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
import torch
import time
import copy
from src.config import device, datasets
import codecs
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from utils import image2tensorboard
from src.opt import opt

record_num = 3
label_dict = datasets[opt.dataset]


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,
                       model_save_path="./", log_save_path=""):
    since = time.time()
    val_acc_history = []
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    log_file_writer = codecs.open(log_save_path, mode="w")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)
        log_file_writer.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        log_file_writer.write('-' * 10 + "\n")

        for name, param in model.named_parameters():
            writer.add_histogram(
                name, param.clone().data.to("cpu").numpy(), epoch)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch_num = 0
            batch_start_time = time.time()
            for names, inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if batch_num % 100 == 0:
                    print("batch num:", batch_num, "cost time:", time.time() - batch_start_time)
                    batch_start_time = time.time()
                batch_num += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            log_file_writer.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_save_path)

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                writer.add_scalar("scalar/val_acc", epoch_acc, epoch)
                writer.add_scalar("Scalar/val_loss", epoch_loss, epoch)
                imgnames, pds = names[:3], [label_dict[i] for i in preds[:record_num].tolist()]
                for idx, (img_path, pd) in enumerate(zip(imgnames, pds)):
                    img = cv2.imread(img_path)
                    img = cv2.putText(img, pd,(20, 50),cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                    #cv2.imwrite("tmp/{}_{}.jpg".format(epoch, idx), img)
                    tb_img = image2tensorboard(img)
                    # images = torch.cat((images, torch.unsqueeze(tb_img, 0)), 0)
                    writer.add_image("pred_image_for_epoch{}".format(epoch), tb_img, epoch)
                # writer.add_image("pred_image_for_epoch{}".format(epoch), images[1:, :, :, :])

            else:
                writer.add_scalar("scalar/train_acc", epoch_acc, epoch)
                writer.add_scalar("Scalar/train_loss", epoch_loss, epoch)

        epoch_time_cost = time.time() - epoch_start_time
        print("epoch complete in {:.0f}m {:.0f}s".format(epoch_time_cost // 60, epoch_time_cost % 60))
        log_file_writer.write(
            "epoch complete in {:.0f}m {:.0f}s\n".format(epoch_time_cost // 60, epoch_time_cost % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    # print('Best test Acc: {:.4f}'.format(max(test_acc_history)))

    if is_inception:
        writer.add_graph(model, torch.rand(1, 3, 299, 299).to(device))
    else:
        writer.add_graph(model, torch.rand(1, 3, 224, 224).to(device))

    log_file_writer.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    log_file_writer.write('Best val Acc: {:.4f}\n'.format(best_acc))
    # log_file_writer.write('Best test Acc: {:.4f}\n'.format(max(test_acc_history)))

    log_file_writer.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
