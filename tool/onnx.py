import numpy as np
import onnxruntime as ort

img = np.zeros([1, 3, 224, 224], dtype=np.float64)
model_onnx = ort.InferenceSession('/media/hkuit164/WD20EJRX/onnx_sim/CNN-CatDog_resnet18_sim.onnx')
outputs = model_onnx.run(None, {'input.1': img})
print(outputs)
