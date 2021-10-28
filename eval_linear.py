import argparse
import os
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from collections import OrderedDict

def get_args_parser():
    parser = argparse.ArgumentParser(description='workflow for self-supervised learning')

    # Training/Optimization parameters
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')

    # Misc
    parser.add_argument('--cuda', default='0', type=str, help='GPU that can be seen by the models')
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
        # /tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC/train
    parser.add_argument('--ckpt_path', default='/path/to/checkpoint/', type=str,
        help='Please specify path to the pretext model checkpoints.')
        # /tmp2/aislab/ckpt/SSL/Epoch_0005.pth
    parser.add_argument('--seed', default=3084, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of data loading workers per GPU.')
    
    return parser

args = get_args_parser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image

import lib.utils

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_linear():
    # ============ setup environment ============
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ============ building network ... ============
    arch = 'resnet50'
    model = models.__dict__[arch]()
    in_feature = model.fc.in_features
    model.fc = nn.Identity()
    ckpt = torch.load(args.ckpt_path)
    model_state_dict = ckpt['model_state_dict']
    model_state_dict = {k.replace("fc.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    model.eval()

    linear_classifier = LinearClassifier(in_feature, num_labels=args.num_labels)
    linear_classifier.to(device)
    linear_classifier.train()

    # ============ preparing data ... ============
    

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

if __name__ == '__main__':
    eval_linear()