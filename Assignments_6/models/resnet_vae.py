import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.ops as ops
from torchvision.models.convnext import LayerNorm2d, CNBlockConfig, Conv2dNormActivation

import torchgadgets as tg

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union, Tuple


"""
    The ResNet18 VAE module was adapted from: https://github.com/julianstastny/VAE-ResNet18-PyTorch.
    We restructured the VAE module and added the functionality to use pre-trained models.
    In addition, we added inverse bottleneck blocks such that we can also construct decoders from ResNet50 and ResNet101.
    So far, we did not test the larger models and concentrated on the ResNet18 which uses simpler encoding/decoding blocks.

"""

def get_resnet_vae(config):
    """
        Function initializes a ResNet-VAE model according to the config
    """
    size = config['model_size']
    pretrained = config['pretrained']
    if size==18:
        block_setting = [2, 2, 2, 2]
        block = BasicBlockEnc
        block_dec = BasicBlockDec
    elif size==34:
        block_setting = [2, 2, 2, 2]
        block = BasicBlockEnc
        block_dec = BasicBlockDec

    elif size==50:
        block_setting = [3, 4, 6, 3]
        block = BottleneckEnc
        block_dec = BottleneckDec
    elif size==101:
        block_setting = [3, 4, 23, 3]
        block = BottleneckEnc
        block_dec = BottleneckDec
    ##-- Build Encoder --##
    if pretrained:
        encoder = tg.models.ResNet(size=size, layer=1 if config['ada_pool'] else 2)
    else:
        encoder = ResNetEnc(num_blocks=block_setting, block=block, z_dim=config['latent_dim'])

    ##-- Build Decoder --##
    decoder = ResNetDec(num_blocks=block_setting, block=block_dec, z_dim=config['latent_dim'])

    vae = BaseVAE(config, encoder, decoder)

    return vae

##-- Convulutional Layers --##
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


##-- Transpose Convolutional Layers--##
# Instead of usuing a transposed convolution we first interpolate and then apply the convolution
def transConv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    
    if stride==1:
        return  nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
    else:
        return ResizeConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            scale_factor=stride
        )
    

def transConv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    if stride==1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return ResizeConv2d(
            in_planes,
            out_planes,
            kernel_size=1,
            scale_factor=stride
        ) 

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x
    
##-- Basic Encoder/Decoder Blocks --##
# These modules are copy+paste from: https://github.com/julianstastny/VAE-ResNet18-PyTorch.

class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, planes, stride=1, upsample=None):
        super().__init__()

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
    
##-- Bottleneck Block --##
# This is the implementation taken from TorchVision
class BottleneckEnc(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
    
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out
    
##-- Inverse Bottleneck Block --##
# Inverse bottleneck blog to be used within the decoder
# Unfortunately, we were not able to test it yet.

class BottleneckDec(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None

    ) -> None:
        super().__init__()

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv3 = transConv1x1(planes * self.expansion, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = transConv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv1 = transConv1x1(planes, in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.upsample = upsample
    
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(x)
        out = self.bn1(out)
        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out
##-- Custom ResNet Encoder Model --##
class ResNetEnc(nn.Module):

    def __init__(self, block=BasicBlockEnc, num_blocks=[2,2,2,2], z_dim=10, nc=3, ada_pool=True):
        super().__init__()
        self.in_planes = 64
        self.ada_pool = ada_pool
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_Blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []
        layers += [block(self.in_planes, planes, stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for i in range(1,num_Blocks):
            layers += [block(self.in_planes, planes)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        #x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.ada_pool:
            x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

##-- ResNet Decoder Model --##
class ResNetDec(nn.Module):

    def __init__(self, block=BasicBlockDec, num_blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512 * block.expansion
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, block, planes, num_Blocks, stride):
        upsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            upsample = nn.Sequential(
                transConv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
     
        layers = []
        layers += [block(self.in_planes, planes, stride, upsample=upsample)]
        self.in_planes = planes * block.expansion
        for i in range(1,num_Blocks):
            layers += [block(self.in_planes, planes)]
        

        return nn.Sequential(*layers)

    def forward(self, z):
        import ipdb; ipdb.set_trace()
        x = self.upsample(z)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 64, 64)
        return x
    

class BaseVAE(nn.Module):
    """
        Base class for a variational autoencoder. The encoder and decoder can be manually defined and passed to the VAE.
        Here we use it to build our ResNet-VAE
    
    """

    def __init__(self, model_config, encoder, decoder):
        super(BaseVAE, self).__init__()
        # Set model config

        self.latent_dim = model_config['latent_dim']
        self.input_size = model_config['input_size']
        self.conv_output = model_config['conv_output']
        self.enc_out = model_config['enc_bottleneck_dim']
        self.conditional = model_config['conditional']
        
        if self.conditional:
            self.embed_class = nn.Linear(model_config['num_classes'], self.input_size[1] * self.input_size[2])
         

        self.encoder = encoder
        self.decoder = decoder

        self.fc_mu = nn.Linear(in_features = self.conv_output[0]*self.conv_output[1]*self.conv_output[2], out_features = self.latent_dim)
        self.fc_sigma= nn.Linear(in_features = self.conv_output[0]*self.conv_output[1]*self.conv_output[2], out_features = self.latent_dim)

        # Define the bottleneck layers, if needed, at the end of the encoder or beginning of decoder
        self.enc_bottleneck = None
        self.dec_bottleneck = None
        if model_config['encoder_bottleneck']: 
            self.enc_bottleneck = nn.Sequential(
                *[
                    nn.Linear(self.conv_output[0]*self.conv_output[1]*self.conv_output[2], self.enc_out),
                    nn.BatchNorm1d(self.enc_out),
                    nn.ReLU()

                ]
            )

        if model_config['decoder_bottleneck']: 
            self.dec_bottleneck = nn.Sequential(
                *[
                    nn.Linear(self.latent_dim if not self.conditional else self.latent_dim+model_config['num_classes'], self.conv_output[0]*self.conv_output[1]*self.conv_output[2]),
                    nn.BatchNorm1d(self.conv_output[0]*self.conv_output[1]*self.conv_output[2]),
                    nn.ReLU()

                ]
            )
        else:
            # Even if we do not want a bottleneck layer we have to project the latent dimension up to the input dimension of the decoder
            self.dec_bottleneck = nn.Linear(self.latent_dim if not self.conditional else self.latent_dim+model_config['num_classes'], self.conv_output[0]*self.conv_output[1]*self.conv_output[2])

    
    def forward(self, x, y=None):
        import ipdb; ipdb.set_trace()
        z, (mu, sigma) = self.encode(x, y)

        x_out = self.decode(z, y)
        x_out = x_out.view(-1, *self.input_size)
        return x_out, (z, mu, sigma)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def encode(self, x, y=None):
        if self.conditional:
            assert y is not None, 'Please provide a condition y'
            embedded_class = self.embed_class(y.float())
            embedded_class = torch.repeat_interleave(embedded_class.view(-1, self.input_size[1], self.input_size[2]).unsqueeze(1), self.input_size[0], 1)
            x = x + embedded_class

        enc = self.encoder(x)
        if self.enc_bottleneck is not None:
            enc = self.enc_bottleneck(enc)
        enc = enc.squeeze()
        mu = self.fc_mu(enc)
        logvar = self.fc_sigma(enc)
        z = self.reparametrize(mu, logvar)
        return z, (mu, logvar)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def decode(self, z, y=None):
        if self.conditional:
            assert y is not None, 'Please provide a condition y'
            z = torch.cat([z,y.float()], dim=-1)
        z = self.dec_bottleneck(z)
        z = z.view(-1, *self.conv_output)
        return self.decoder(z)
