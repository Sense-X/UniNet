import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def to4d(x):
    if len(x.shape) == 4:
        return x
    B, N, C = x.shape
    h = int(N ** 0.5)
    return x.transpose(1, 2).reshape(B, C, h, h)


def to3d(x):
    if len(x.shape) == 3:
        return x
    B, C, h, w = x.shape
    N = h * w
    return x.reshape(B, C, N).transpose(1, 2)


class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input, inplace):
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        # return F.hardsigmoid(scale, inplace=inplace)
        return hard_sigmoid(scale, inplace=inplace)

    def forward(self, input):
        scale = self._scale(input, True)
        return scale * input


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6

