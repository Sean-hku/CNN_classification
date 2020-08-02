# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import argparse

parser = argparse.ArgumentParser(description='PyTorch CNN Training')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--expFolder', default='test', type=str,
                    help='Experiment folder')
parser.add_argument('--nThreads', default=30, type=int,
                    help='Number of data loading threads')
parser.add_argument('--dataset', default='ceiling', type=str,
                    help='Experiment folder')

"----------------------------- Model options -----------------------------"
parser.add_argument('--backbone', default="mobilenet", type=str,
                    help='The backbone of the model')
parser.add_argument('--loadModel', default=None, type=str,
                    help='Provide full path to a previously trained model')

"----------------------------- Hyperparameter options -----------------------------"
parser.add_argument('--LR', default=1e-3, type=float,
                    help='Learning rate')
parser.add_argument('--momentum', default=0, type=float,
                    help='Momentum')
parser.add_argument('--weightDecay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='epsilon')
parser.add_argument('--crit', default='MSE', type=str,
                    help='Criterion type')
parser.add_argument('--freeze', default=0, type=float,
                    help='freeze backbone')
parser.add_argument('--freeze_bn', default=False, type=bool,
                    help='freeze bn')
parser.add_argument('--optMethod', default='adam', type=str,
                    help='Optimization method: rmsprop | sgd | nag | adadelta')
parser.add_argument('--sparse_s', default=0, type=float,
                    help='sparse')
parser.add_argument('--sparse_decay', default=1, type=float,
                    help='sparse_decay')


"----------------------------- Training options -----------------------------"
parser.add_argument('--epoch', default=20, type=int,
                    help='Total number to train')
parser.add_argument('--val_ratio', default=0.3, type=int,
                    help='Current epoch')
parser.add_argument('--batch', default=12, type=int,
                    help='Train-batch size')
parser.add_argument('--trainIters', default=0, type=int,
                    help='Total train iters')
parser.add_argument('--valIters', default=0, type=int,
                    help='Total valid iters')
parser.add_argument('--num_worker', default=5, type=int,
                    help='num worker of train')
parser.add_argument('--save_interval', default=1, type=int,
                    help='interval')

opt = parser.parse_args()
