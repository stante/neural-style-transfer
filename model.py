import torch.nn as nn
import torchvision.models as models


class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super(NeuralStyleTransfer, self).__init__()
        self.model = models.vgg19(pretrained=True).features

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.style_weights = {'conv1_1': 1.0,
                              'conv2_1': 0.8,
                              'conv3_1': 0.5,
                              'conv4_1': 0.3,
                              'conv5_1': 0.1}

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
