from byol_pytorch import BYOL
from torchvision import models, transforms
from torch import nn
import torch.nn.functional as F
import torch

class byol(nn.Module):
    """
    Double Conv for U-Net
    """
    def __init__(self, load = False, path = '/home1/qiuliwang/Code/byol-pytorch-master/byol_pytorch/checkpoint/500.pth.tar'):
        super(byol, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.model = BYOL(
            resnet,
            image_size = 256,
            hidden_layer = 'avgpool',
            projection_size = 1024,
            projection_hidden_size = 4096,
            moving_average_decay = 0.99
        )

        if load:
            print('Loading...', path)
            self.model.load_state_dict(torch.load(path), strict = False)
        else:
            self.model.load_state_dict(torch.load('/home1/qiuliwang/Code/byol-pytorch-master/byol_pytorch/checkpoint/500.pth.tar'), strict = False)

        # self.fc1 = nn.Linear(2048, 512)
        # self.fc2 = nn.Linear(512, 2)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, 64)

    def forward(self, x):
        projection, representation = self.model(x, return_embedding = True)

        # x = representation
        # x1 = self.fc1(x)
        # x2 = self.fc2(x1)
        # x3 = self.fc2(x2)
        # x4 = self.fc2(x3)
        # x5 = self.fc2(x4)
        return projection, representation
