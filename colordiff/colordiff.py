import torch

def srgb2xyz(images: torch.Tensor) -> torch.Tensor:
    """ Convert Standard-RGB to XYZ

    Args:
        images: A torch tensor with shape N x H x W x C containing standard-RGB
          values from [0, 1].

    Returns:
        A torch tensor with the same shape as the input, containing XYZ 
          representations of the input images.
    """
    mask = (images > 0.04045).float()
    images = 100 * (mask * (((images + 0.055) / 1.055) ** 2.4) + 
                    (1 - mask) * (images / 12.92))
    
    mat = torch.tensor([
        [0.4124, 0.2126, 0.0193],
        [0.3576, 0.7152, 0.1192],
        [0.1805, 0.0722, 0.9505],
    ])

    images = torch.matmul(images, mat)
    return images

def xyz2lab(images: torch.Tensor) -> torch.Tensor:
    """ Convert XYZ color space to CIE-Lab

    Args:
        images: A torch tensor with shape N x H x W x C containing XYZ
          representation of the images.
    
    Returns:
        A torch tensor with the same shape as the input, containing CIE-Lab
          representation of the input images.
    """
    ref_xyz = torch.tensor([95.047, 100.000, 108.883])

    images /= ref_xyz
    mask = (images > 0.008856).float()
    images = ((mask) * (images ** (1. / 3.)) + 
                (1 - mask) * ((7.787 * images) + (16. / 116.)))
    
    mat = torch.tensor([
        [  0.,  500.,    0.],
        [116., -500.,  200.],
        [  0.,    0., -200.],
    ])

    bias = torch.tensor([-16., 0., 0.])
    images = torch.matmul(images, mat) + bias
    return images






     