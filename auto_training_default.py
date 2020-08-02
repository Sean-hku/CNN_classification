cmds = [
    'python train_model.py --backbone mnasnet --batch 32 --epoch 20--expFolder mnasnet --expID 32',
    'python train_model.py --backbone mnasnet --batch 16 --epoch 20 --expFolder mnasnet --expID 16',
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)

