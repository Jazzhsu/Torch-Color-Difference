import torch
from colordiff import colordiff

# N x C x H x w
im = torch.tensor([[[[0.01, 0.01]], [[0.6, 0.6]], [[0.02, 0.02]]]])

# NCHW -> NHWC
im = im.permute(0, 2, 3, 1)

print(colordiff.xyz2lab(colordiff.srgb2xyz(im)))