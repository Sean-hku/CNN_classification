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
## Draw P-R curve
you should modify `model_path`, `img_path`, `classes` in src/config and `batch` in src/opt, run:
```bash
python test.py
```
## Sparse Training
```bash
python train_model.py --dataset ceiling --backbone resnet18 --sparse
```
## Normal Pruning(Resnet18)
you should modify the `model_path`,  `save_cfg_path`, `save_model_path` in prune_config and then run:
```bash
python prune/pruning.py 
```
if you want to test the pruned model, you can add `test_model_path` and `img_path` in prune_config.
