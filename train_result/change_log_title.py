import os
src_path = '../weight/ceiling_action-7_class'

for folder in os.listdir(src_path):
    log_path = os.path.join(src_path, folder, folder, "train_log.csv")
    with open(log_path) as log:
        lines = [line for line in log.readlines()]

