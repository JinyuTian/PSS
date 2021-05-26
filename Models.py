import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from l0_layers import L0Conv2d
import torch
import torch.nn.init as init

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            L0Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                L0Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            L0Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        # self._initialize_weights()
        self.layers = layers
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out


class OriDnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()
        self.layers = layers
    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class MyVGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(MyVGG, self).__init__()
        self.features = features
        self.layers = []
        for m in self.features:
            if isinstance(m, nn.Linear) or isinstance(m, L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear) or isinstance(m, L0Conv2d):
                self.layers.append(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                self.bn_params += [m.weight, m.bias]

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def regularization(self):
        ReconstructionError = 0.
        for layer in self.layers:
            if hasattr(layer,'qz_loga'):
                if layer.qz_loga.shape == torch.Size([64]):
                    p = 0.0025
                elif layer.qz_loga.shape == torch.Size([128]):
                    p = 0.3
                elif layer.qz_loga.shape == torch.Size([256]):
                    p = 0.3
                elif layer.qz_loga.shape == torch.Size([512]):
                    p = 0.3
                ReconstructionError += p*layer.ReconstructionError()
        return ReconstructionError.cuda()

    def soft_regularization(self):
        ReconstructionError = 0.
        for layer in self.layers:
            if hasattr(layer,'qz_loga'):
                if layer.qz_loga.shape == torch.Size([64]):
                    p = 0.00001
                elif layer.qz_loga.shape == torch.Size([128]):
                    p = 0.003
                elif layer.qz_loga.shape == torch.Size([256]):
                    p = 0.003
                elif layer.qz_loga.shape == torch.Size([512]):
                    p = 0.003
                ReconstructionError += p*layer.soft_regularization()
        return ReconstructionError.cuda()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            L0Conv = L0Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=True)
            if batch_norm:
                layers += [L0Conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [L0Conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg19(pretrained=False,**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = MyVGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model

