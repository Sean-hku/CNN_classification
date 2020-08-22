cmds = [
    'python train_model.py --backbone mnasnet --batch 32 --epoch 20 --expFolder mnasnet --expID 32 --dataset ceiling',
    'python train_model.py --backbone mnasnet --batch 16 --epoch 20 --expFolder mnasnet --expID 16 --dataset ceiling',
]

import os
log = open("train_log.log", "a+")
for idx, cmd in enumerate(cmds):
    log.write(cmd)
    log.write("\n")
    print("Processing cmd {}".format(idx))
    os.system(cmd)

