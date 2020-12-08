# CNN classification
## Requiremets <br>
python 3.6 or later with all requirements.txt dependencies installed, including `torch>=1.4`. To install run:
```bash
$ pip install -r requirements.txt
````

## Quick demo:
you should add the `model path` and `img path` in the `src/config` and the run:
```bash
python test_model.py 
```

## Training
```bash
python train_model.py --dataset ceiling --backbone resnet18
```

## GradCAM
you should add the `model path` , `img path` , `target_layer` in the run_CAM.py and then run:
```bash
python Grad_CAM/run_CAM.py 
```
