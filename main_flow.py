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
    parser.add_argument('--aux_labels', default=512, type=int, help='Number of auxiliary labels')

    # Misc
    parser.add_argument('--cuda', default='0', type=str, help='GPU that can be seen by the models')
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
        # /tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC/train
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

def train_auxiliary():
    # ============ setup environment ============
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ============ preparing data ... ============
    imagenet_size = 224
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)
    crops_scale   = (0.4, 1.0)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(imagenet_size, scale=crops_scale, interpolation=Image.BICUBIC), 
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_per_gpu, \
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,)
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ preparing auxiliary label ... ============
    loom = Loom(imagenet_size, args.batch_size_per_gpu, args.aux_labels).to(device)
    distortions = loom.get_distortions(alpha=1024, epsilon=5, kernel_size=51)

    # ============ building network ... ============
    arch = 'resnet50'
    model = models.__dict__[arch]()
    # add projection head
    in_feature = model.fc.in_features
    out_feature = args.aux_labels # total number of distortions
    model.fc = nn.Sequential(OrderedDict([
        ('aux_fc', nn.Linear(in_feature, out_feature)),
    ]))
    model.to(device)
    
    # ============ preparing loss and optimizer ============
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # ============ training ... ============
    def train():
        train_loss, correct, total = 0, 0, 0
        model.train()

        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(device)
            # make label and distortion grid
            aux_label = torch.randint(0, args.aux_labels, label.shape).to(device)
            aux_grid = torch.empty(0, 2, data.shape[2], data.shape[3]).to(device)
            for i in aux_label:
                aux_grid = torch.cat((aux_grid, distortions[i:i+1,:,:,:]), 0)
            # distortion
            data = loom(data, aux_grid)

            image_dir = Path('./picture') / 'p'
            image_dir.mkdir(parents=True, exist_ok=True)
            normal_path = image_dir / 'examples_normal.png'
            # use the torchvision save_image method
            if batch_idx==0:
                save_image(data[0:5], str(normal_path))

            # train the model
            optimizer.zero_grad()
            logit = model(data)
            loss = criterion(logit, aux_label)
            loss.backward()
            optimizer.step()
            preds = F.softmax(logit, dim=1)
            preds_top_p, preds_top_class = preds.topk(1, dim=1)
            train_loss += loss.item() * aux_label.size(0)
            total += aux_label.size(0)
            correct += (preds_top_class.view(aux_label.shape) == aux_label).sum().item()

            if batch_idx > 100:
                print('==> early break in training')
                break
            
        return (train_loss / batch_idx, 100. * correct / total)

    start_time = time.time()
    print("Starting auxiliary training !")
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train()
        print('train loss:{:.4f}, train acc:{:.4f}'.format(train_loss, train_acc))

        # ============ writing logs ... ============
        ckpt_dir = Path('/tmp2/aislab/ckpt') / 'SSL'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / (f'Epoch_{epoch:04}.pth')
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        torch.save(save_dict, ckpt_path)
    total_time = time.time() - start_time
    print(f'Training time {total_time/60:.2f} minutes')





class Loom(nn.Module):
    def __init__(self, image_size, batch_size, aux_labels):
        super(Loom, self).__init__()
        self.image_size, self.aux_labels = image_size, aux_labels
        base_grid = self.get_base_grid(batch_size)
        self.register_buffer('base_grid', base_grid)

    def get_base_grid(self, batch_size):
        # output shape (N, 2. H, W)
        sequence = torch.arange(-(self.image_size-1), (self.image_size), 2)/(self.image_size-1.0)
        samp_grid_x = sequence.repeat(self.image_size,1)
        samp_grid_y = samp_grid_x.t()
        samp_grid = torch.cat((samp_grid_x.unsqueeze(0), samp_grid_y.unsqueeze(0)), 0)
        base_grid = samp_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return base_grid

    def get_distortions(self, alpha, epsilon, kernel_size, sigma=None):
        # generate random field, double blur to create larger blob in the image
        value_range = (2.0/self.image_size)*alpha
        # first pass
        grid = torch.zeros(self.aux_labels, 2, self.image_size, self.image_size).detach().to(device)
        grid = grid + torch.empty_like(grid).uniform_(-value_range, value_range)
        grid = self.gaussian_blur(grid, kernel_size, sigma)
        # second pass
        pixel_width = 2.0/(self.image_size)
        grid = torch.clamp(grid, -pixel_width*epsilon, pixel_width*epsilon)
        grid = self.gaussian_blur(grid, kernel_size, sigma)
        return grid

    def get_gaussian_kernel2d(self, kernel_size, sigma=None):
        if sigma == None:
            sigma = kernel_size*0.15 + 0.35
        ksize_half = (kernel_size-1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        kernel2d = torch.mm(kernel1d[:, None], kernel1d[None, :])
        return kernel2d

    def gaussian_blur(self, grid, kernel_size, sigma=None):
        # kernel_size should have odd and positive integers
        kernel = self.get_gaussian_kernel2d(kernel_size, sigma)
        kernel = kernel.expand(grid.shape[-3], 1, kernel.shape[0], kernel.shape[1]).to(device)
        # padding = (left, right, top, bottom)
        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
        grid = F.pad(grid, padding, mode="reflect")
        grid = F.conv2d(grid, kernel, groups=grid.shape[-3])
        return grid

    def forward(self, image, grid):
        samp_grid = grid + self.base_grid
        binding_grid = samp_grid.permute(0,2,3,1)
        distort_image = F.grid_sample(image, binding_grid, align_corners=True)
        return distort_image


if __name__ == '__main__':
    train_auxiliary()