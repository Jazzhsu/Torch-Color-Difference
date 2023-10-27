import torch
from torch_de import delta_e

# N x C x H x W
im = torch.tensor([[[[100., 100.]], [[21.57210357, 21.57210357]], [[272.22819350, 272.22819350]]]])
im2 = torch.tensor([[[[100., 50.]], [[426.67945353, 426.67945353]], [[72.39590835, 72.39590835]]]])

# NCHW -> NHWC
im = im.permute(0, 2, 3, 1)
im2 = im2.permute(0, 2, 3, 1)

# 94.0356, 100.8779
print(delta_e.ciede2000_lab(im, im2))

