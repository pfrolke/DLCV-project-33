import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGG19Style(nn.Module):
    def __init__(self, num_attributes):
        super().__init__()

        # select 16 convolutional and 5 pooling layers

        layers = vgg19(weights=VGG19_Weights.DEFAULT).features

        for i, layer in enumerate(layers):
            if isinstance(layer, nn.MaxPool2d):
                layers[i] = nn.AvgPool2d(
                    kernel_size=2, stride=2, padding=0, ceil_mode=False
                )

        layers.append(nn.Linear(layers[-1]))

        self.layers = nn.Sequential(layers)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.layers(x)


class VGG19Style2(nn.Module):
    def __init__(self, num_attributes):
        super().__init__()

        # select 16 convolutional and 5 pooling layers

        self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)
        self.classif = nn.Sequential(nn.Linear(1000, 32), nn.Sigmoid())

        for param in self.vgg19.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.vgg19(x)
        x = self.classif(x)
        return x
