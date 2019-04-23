import torch.nn as nn
import torchvision.models as models


class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super(NeuralStyleTransfer, self).__init__()
        self.model = models.vgg16(pretrained=True).features

        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.model(x)
