import torch

def _srgb2xyz(images: torch.Tensor, device = None) -> torch.Tensor:
    """ Convert Standard-RGB to XYZ

    Args:
        images: A torch tensor with shape N x H x W x C containing standard-RGB
          values from [0, 1].
        device: The device where the calculation takes place.

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
    ], deivce=device)

    images = torch.matmul(images, mat)
    return images

def _xyz2lab(images: torch.Tensor, device = None) -> torch.Tensor:
    """ Convert XYZ color space to CIE-Lab

    Args:
        images: A torch tensor with shape N x H x W x C containing XYZ
          representation of the images.
        device: The device where the calculation takes place.

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
    ], device=device)

    bias = torch.tensor([-16., 0., 0.], device=device)
    images = torch.matmul(images, mat) + bias
    return images

def _srgb2lab(image: torch.Tensor, device = None) -> torch.Tensor:
    return _xyz2lab(_srgb2xyz(image, device), device)

def _to_degree(x: torch.Tensor) -> torch.Tensor:
    return x * 180. / torch.pi

def _to_radians(x: torch.Tensor) -> torch.Tensor:
    return x * torch.pi / 180.

def ciede2000_rgb(
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        device = None,
        channel_first = True
) -> torch.Tensor:
    """
    Calculate CIEDE2000 Delta-E between two sets of Standard-RGB images. Note
    that the RGB value should be ranged from [0, 1].

    Args:
        image_1: NCHW or NHWC shaped image sets.
        image_2: NCHW or NHWC shaped image sets.
        device: The device that the tensors should calculating on.
        channel_first: True if the inputs are NCHW shaped, false if inputs are
          NHWC shaped.

    Returns:
        Color difference between the two sets of images.
    """
    if channel_first:
        image_1 = image_1.permute(0, 2, 3, 1)
        image_2 = image_2.permute(0, 2, 3, 1)

    image_1 = _srgb2lab(image_1, device)
    image_2 = _srgb2lab(image_2, device)
    return ciede2000_lab(image_1, image_2)

def ciede2000_lab(image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
    """ Calculate CIEDE2000 Delta-E between two sets of Lab color space images.

    Args:
        image_1: NHWC shaped torch tensor images with CIE-Lab color space.
        image_2: NHWC shaped torch tensor images with CIE-Lab color space.

    Returns:
        Color differences between the two sets of images.
    """
    l1 = image_1[:, :, :, 0]
    a1 = image_1[:, :, :, 1]
    b1 = image_1[:, :, :, 2]

    l2 = image_2[:, :, :, 0]
    a2 = image_2[:, :, :, 1]
    b2 = image_2[:, :, :, 2]

    k_l = 1
    k_c = 1
    k_h = 1

    c1 = torch.sqrt(a1**2 + b1**2)
    c2 = torch.sqrt(a2**2 + b2**2)
    c_bar = (c1 + c2) / 2.

    c_bar_7 = c_bar**7
    g = 0.5 * (1 - torch.sqrt(c_bar_7 / (c_bar_7 + 25**7)))
    a1_prime = (1 + g) * a1
    a2_prime = (1 + g) * a2

    c1_prime = torch.sqrt(a1_prime**2 + b1**2)
    c2_prime = torch.sqrt(a2_prime**2 + b2**2)

    h1_prime = _to_degree(torch.atan2(b1, a1_prime))
    h2_prime = _to_degree(torch.atan2(b2, a2_prime))

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime

    delta_h_prime = h2_prime - h1_prime
    mask1 = (delta_h_prime > 180).float()
    mask2 = (delta_h_prime < -180).float()

    delta_h_prime = ((1 - mask1 - mask2) * delta_h_prime
                     + mask1 * (delta_h_prime - 360)
                     + mask2 * (delta_h_prime + 360))

    delta_h_prime = (2 * torch.sqrt(c1_prime * c2_prime)
                       * torch.sin(_to_radians(delta_h_prime / 2.)))

    l_prime_bar = (l1 + l2) / 2.
    c_prime_bar = (c1_prime + c2_prime) / 2.

    mask1 = (torch.abs(h1_prime - h2_prime) <= 180).float()
    mask2 = (h1_prime + h2_prime < 360).float()

    h_prime_bar = (h1_prime + h2_prime) / 2.
    h_prime_bar = ((1 - mask1 - mask2) * (h_prime_bar - 180.)
                   + mask1 * h_prime_bar
                   + mask2 * (h_prime_bar + 180.))

    t = (1 - 0.17 * torch.cos(_to_radians(h_prime_bar - 30))
           + 0.24 * torch.cos(_to_radians(2 * h_prime_bar))
           + 0.32 * torch.cos(_to_radians(3 * h_prime_bar + 6))
           - 0.20 * torch.cos(_to_radians(4 * h_prime_bar - 63)))

    delta_theta = 30 * torch.exp(-((h_prime_bar - 275) / 25)**2)

    c_prime_bar_7 = c_prime_bar**7
    r_c = 2 * torch.sqrt(c_prime_bar_7 / (c_prime_bar_7 + 25**7))
    s_l = 1 + ((0.015 * (l_prime_bar - 50)**2)
              / torch.sqrt(20 + (l_prime_bar - 50)**2))
    s_c = 1 + 0.045 * c_prime_bar
    s_h = 1 + 0.015 * c_prime_bar * t
    r_t = -torch.sin(2 * delta_theta) * r_c

    delta_e = torch.sqrt(
        (delta_l_prime / (k_l * s_l)) ** 2
      + (delta_c_prime / (k_c * s_c)) ** 2
      + (delta_h_prime / (k_h * s_h)) ** 2
      + r_t * (delta_c_prime / (k_c * s_c)) * (delta_h_prime / (k_h * s_h))
    )

    return delta_e