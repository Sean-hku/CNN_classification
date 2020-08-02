cmds = [
    'python train_model.py --dataset ceiling --backbone mobilenet --batch 32 --epoch 20 --LR 1.00E-03 --expFolder ceiling_2cls --expID 1',
    'python train_model.py --dataset ceiling --backbone mnasnet --batch 32 --epoch 20 --LR 1.00E-03 --expFolder ceiling_2cls --expID 2',
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)

