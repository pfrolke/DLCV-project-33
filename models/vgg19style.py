import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGG19Style(nn.Module):
    def __init__(self):
        super().__init__()

        # select 16 convolutional and 5 pooling layers

        self.layers = vgg19(weights=VGG19_Weights.DEFAULT).features

        # swap maxpool layers for avgpool
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxPool2d):
                self.layers[i] = nn.AvgPool2d(
                    kernel_size=2, stride=2, padding=0, ceil_mode=False
                )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, ):
        return self.layers(x)
