from src.tester import ModelInference
import os
import cv2

num_classes = 1000


class Tester:
    def __init__(self, model_path):
        self.pre_name = self.__get_pretrain(model_path)
        self.model = ModelInference(num_classes, self.pre_name, model_path)

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
        else:
            raise ValueError("Wrong name of pre-train model")
        return name

    def test_score(self, img):
        score = self.model.predict(img)
        # max_val, max_idx = torch.max(score, 1)
        return score

    def test_idx(self, img):
        score = self.test_score(img)
        idx = score[0].tolist().index(max(score[0].tolist()))
        return idx


if __name__ == '__main__':
    model_pth = "models/pre_train_model/mnasnet.pth"
    img_path = "tmp/cat.jpeg"
    MI = Tester(model_pth)
    max_idx = MI.test_idx(cv2.imread(img_path))
    print(max_idx)
