import torch.nn as nn
from torch.nn import Sequential

class Flatten(nn.Module):
    """
    Implementing a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        a, b = list(x.size())[:2]
        return x.view(a, b)

def all_cnn_module():
    """
    Creating a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Using a AvgPool2d to pool and then Flatten layer as final layers.
    Exactly 23 layers in total, of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """
    layers = [
        nn.Dropout(p=0.2),
        nn.Conv2d(3, 96, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(96, 96, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(96, 96, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Conv2d(96, 192, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(192, 192, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(192, 192, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Conv2d(192, 192, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(192, 192, 1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(192, 10, 1, stride=1, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(6),
        Flatten()
    ]
    return Sequential(*layers)
