from src.tester import ModelInference
import os
import cv2
from src import config
from src.dataloader import DataLoader
classes = ["cat", "dog"]
num_classes = len(classes)
colors = {"cat": (0,255,255), "dog":(255,255,0)}


class Tester:
    def __init__(self, model_path,conf):
        self.pre_name = self.__get_pretrain(model_path)
        self.model = ModelInference(num_classes, self.pre_name, model_path)
        self.conf = conf

    def __get_pretrain(self, model_path):
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

    def test_score(self, img):
        score = self.model.predict(img)
        # max_val, max_idx = torch.max(score, 1)
        return score

    def test_idx(self, img):
        score = self.test_score(img)
        if list(score[0])[0] > self.conf:
            idx = 0
        else:
            idx = 1
        # idx = score[0].tolist().index(max(score[0].tolist()))
        return idx

    def test_pred(self, img):
        idx = self.test_idx(img)
        pred = classes[idx]
        return pred

    def show_img(self, img, pred):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, pred, (50, 50), font, 2, colors[pred], 3)
        cv2.imshow("Result", img)
        cv2.waitKey(0)

    def to_onnx(self):
        self.model.to_onnx()


if __name__ == '__main__':
    model_pth = config.test_model_path
    img_path = config.test_img
    batch_size = 16
    data_loader = DataLoader(img_path, batch_size)
    for names, inputs, labels in data_loader.dataloaders_dict['val']:
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")
    MI = Tester(model_pth)
    # max_idx = MI.test_idx(cv2.imread(img_path))
    # print(max_idx)
    for img_name in os.listdir(img_path):
        im = cv2.imread(os.path.join(img_path, img_name))
        pred = MI.test_pred(im)
        # MI.show_img(im, pred)
        # print("Prediction of {} is {}".format(img_name, pred))
    # MI.to_onnx()

