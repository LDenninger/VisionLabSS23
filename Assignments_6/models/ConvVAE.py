import torch
import torch.nn as nn
import torchvision as tv
import torchvision.ops as ops
from torchvision.models.convnext import LayerNorm2d, CNBlockConfig, Conv2dNormActivation

import torchgadgets as tg

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union, Tuple


###--- Convolutional Variational Autoencoder ---###

class ConvVAE(nn.Module):
    """
        Base class for a variational autoencoder.
    
    """

    def __init__(self, model_config):
        super(ConvVAE, self).__init__()
        # Set model config
        self.input_size = model_config['input_size']
        self.kernel_size = model_config['kernel_size']
        self.activation = model_config['activation']    
        self.batch_norm = model_config['batch_norm']  
        self.hidden_dim = model_config['hidden_dim']
        self.latent_dim = model_config['latent_dim']
        self.conv_output = model_config['conv_output']
        self.fc_input = model_config['fc_input']
        self.fc_output = model_config['fc_output']
        self.pooling_layers = model_config['pooling_layers']

        self._build_encoder()
        self._build_decoder()


    
    def forward(self, x, y=None):
        z, (mu, sigma) = self.encode(x, y)
        x_out = self.decode(z)
        x_out = x_out.view(-1, *self.input_size)
        return x_out, (z, mu, sigma)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def encode(self, x, y=None):
        enc = self.encoder(x)
        if y is not None:
            enc = torch.cat([enc, y], dim=-1)
        
        mu = self.fc_mu(enc)
        sigma = self.fc_sigma(enc)

        z = self.reparametrize(mu, sigma)

        return z, (mu, sigma)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def decode(self, z, y=None):
        if y is not None:
            z = torch.cat([z, y], dim=-1)
        return self.decoder(z)
    
    def _build_encoder(self):
        layers = []
        conv_dims = [self.input_size[0]] + self.hidden_dim
        
        for i in range(len(conv_dims)-1):
            layers.append(EncoderBlock(in_channels=conv_dims[i],
                                            out_channels=conv_dims[i+1],
                                                kernel_size=self.kernel_size,
                                                    stride=1,
                                                        padding=self.kernel_size[0]//2,
                                                            activation=self.activation))
            if self.pooling_layers[i]:
                layers.append(nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)))

        layers.append(nn.Flatten())
        # Bottleneck layer
        layers.append(nn.Linear(self.fc_input, self.fc_output))
        layers.append(nn.BatchNorm1d(self.fc_output, momentum=0.9))
        layers.append(self._get_activation())

        self.fc_mu = nn.Linear(self.fc_output, self.latent_dim)
        self.fc_sigma = nn.Linear(self.fc_output, self.latent_dim)

        self.encoder = nn.Sequential(*layers)
        
    
    def _build_decoder(self):
        layers = []
        conv_dims =self.hidden_dim[::-1] + [self.input_size[0]]
        pooling_layers = self.pooling_layers[::-1] 

        layers.append(nn.Linear(self.latent_dim, self.fc_input))
        layers.append(nn.BatchNorm1d(self.fc_input, momentum=0.9))
        layers.append(self._get_activation())
        layers.append(tg.models.Reshape(shape=[-1]+self.conv_output))
        
        for i in range(len(conv_dims)-1):
            if pooling_layers[i]:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            layers.append(DecoderBlock(in_channels=conv_dims[i],
                                            out_channels=conv_dims[i+1],
                                                kernel_size=self.kernel_size,
                                                    stride=1,
                                                        padding=self.kernel_size[0]//2,
                                                            activation=self.activation if i<len(conv_dims)-2 else 'sigmoid'))
            
        self.decoder = nn.Sequential(*layers)

    def _get_activation(self):
        if self.activation=='relu':
            return nn.ReLU()
        if self.activation=='gelu':
            return nn.GELU()
        if self.activation=='leaky_relu':
            return nn.LeakyReLU()
        if self.activation=='sigmoid':
            return nn.Sigmoid()

class EncoderBlock(nn.Module):
    def __init__(self,
                    in_channels,
                        out_channels,
                            kernel_size,
                                stride,
                                    padding,
                                        batch_norm=True,
                                            activation='relu'):
        super(EncoderBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.9))

        if activation=='relu':
            layers.append(nn.ReLU())
        if activation=='gelu':
            layers.append(nn.GELU())
        if activation=='leaky_relu':
            layers.append(nn.LeakyReLU())
        if activation=='sigmoid':
            layers.append(nn.Sigmoid())

        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        return self.enc(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                    in_channels,
                        out_channels,
                            kernel_size,
                                stride,
                                    padding,
                                        batch_norm=True,
                                            activation='relu'):
        super(DecoderBlock, self).__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels, momentum=0.9))

        if activation=='relu':
            layers.append(nn.ReLU())
        if activation=='gelu':
            layers.append(nn.GELU())
        if activation=='leaky_relu':
            layers.append(nn.LeakyReLU())
        if activation=='sigmoid':
            layers.append(nn.Sigmoid())

        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        return self.enc(x)
    