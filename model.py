import torch.nn as nn
import torchvision.models as models


class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super(NeuralStyleTransfer, self).__init__()
        self.model = models.vgg19(pretrained=True).features

        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1',
                  '21': 'conv4_2'}
        features = {}
        for name, module in self.model.named_children():
            x = module(x)
            if name in layers:
                features[layers[name]] = x

        return features
