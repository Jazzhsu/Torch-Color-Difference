import pytest
import torch
from torch_de import delta_e

def test_images_with_different_shapes():
    # Test that ValueError is correctly raised when input dimensions of the two
    # image sets do not match.
    image_1 = torch.ones((1, 3, 3, 4))
    image_2 = torch.ones((1, 3, 3, 5))

    with pytest.raises(ValueError):
        delta_e.ciede2000_rgb(image_1, image_2)

def test_images_with_wrong_shape():
    # Test that ValueError is correctly raised when input dimension is not
    # correct.
    image_1 = torch.ones((1, 3, 3))
    image_2 = torch.ones((1, 3, 3))

    with pytest.raises(ValueError):
        delta_e.ciede2000_rgb(image_1, image_2)

def test_images_with_wrong_channel_size():
    # Test that when |channel_first| is False, ValueError is correctly raised
    # when the last dimension is not size 3.
    image_1 = torch.ones((1, 3, 5, 5))
    image_2 = torch.ones((1, 3, 5, 5))

    with pytest.raises(ValueError):
        delta_e.ciede2000_rgb(image_1, image_2, channel_first=False)

    # Test that when |channel_first| is True, ValueError is correctly raised
    # when the second dimension is not size 3.
    image_1 = torch.ones((1, 5, 5, 3))
    image_2 = torch.ones((1, 5, 5, 3))

    with pytest.raises(ValueError):
        delta_e.ciede2000_rgb(image_1, image_2, channel_first=True)

def test_ciede2000_lab():
    # Test the correctness of the calculation.
    image_1 = torch.tensor(
        [[[[100., 21.57210357, 272.22819350],
           [100., 21.57210357, 272.22819350]]]])
    image_2 = torch.tensor(
        [[[[100., 426.67945353, 72.39590835],
           [50., 426.67945353, 72.39590835]]]])

    assert torch.allclose(delta_e.ciede2000_lab(image_1, image_2),
                          torch.tensor([[[94.0356, 100.8779]]]))