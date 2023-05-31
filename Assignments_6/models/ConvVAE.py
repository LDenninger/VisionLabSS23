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

    def __init__(self, input_size: tuple[int],
                         encoder_layers: list[dict], 
                            decoder_layers: list[dict],
                                latent_dim: tuple[int]):
        super(ConvVAE, self).__init__()
        self.input_size = input_size        
        modules= tg.models.build_model(encoder_layers)
        self.encoder = nn.Sequential(*modules)
        modules = tg.models.build_model(decoder_layers)
        self.decoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(latent_dim[0], latent_dim[1])
        self.fc_sigma = nn.Linear(latent_dim[0], latent_dim[1])
    
    def forward(self, x):
        z = self.encode(x)
        x_out = self.decode(z)
        x_out = x_out.view(-1, *self.input_size)
        return x_out

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def encode(self, x):
        self.encoder(x)
        
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        z = self.reparametrize(mu, sigma)

        return z

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def decode(self, z):
        return self.decoder(z)






###--- ConvNeXt Variational Autoencoder ---###

class ConvNeXtVAE(nn.Module):
    """
        Base class for a variational autoencoder.
    
    """

    def __init__(self, input_size: tuple[int],
                         encoder_layers: list[dict], 
                            decoder_layers: list[dict],
                                latent_dim: tuple[int]):
        super(ConvNeXtVAE, self).__init__()

        self.input_size = input_size        
        self.encoder = tg.models.build_model(encoder_layers)
        self.decoder = tg.models.build_model(decoder_layers)

        self.fc_mu = nn.Linear(latent_dim[0], latent_dim[1])
        self.fc_sigma = nn.Linear(latent_dim[0], latent_dim[1])
    
    def forward(self, x):
        z = self.encode(x)
        x_out = self.decode(z)
        x_out = x_out.view(-1, *self.input_size)
        return x_out

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def encode(self, x):
        self.encoder(x)
        
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        z = self.reparametrize(mu, sigma)

        return z

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def decode(self, z):
        return self.decoder(z)
    
###--- ConvNeXt Implementation ---###
# This implementation was taken from: https://github.com/pytorch/pytorch
    
def load_inverse_convnext_model(size):
    # Default ConvNeXt architectures
    inverse_convnext_setup = {
        "tiny": [
                [
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 9),
                CNBlockConfig(768, None, 3),
                ], 0.1],
        "small": [
                [
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 27),
                CNBlockConfig(768, None, 3),
                ], 0.4],
        "base": [
                [
                CNBlockConfig(128, 256, 3),
                CNBlockConfig(256, 512, 3),
                CNBlockConfig(512, 1024, 27),
                CNBlockConfig(1024, None, 3),
                ], 0.5],

        "large": [
                [
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, 768, 3),
                CNBlockConfig(768, 1536, 27),
                CNBlockConfig(1536, None, 3),
                ], 0.5]   
    }
    # Inverse architecture for the decoder
    for k, v in inverse_convnext_setup.items():
        v[1] = v[1].reverse()


    return convnext_setup[size]



###--- Inverse ConvNeXt Implementation ---###
# Custom inversed ConvNeXt model. The model is largely copied from the original implementation to prevent any bugs regarding the inverse architecture.

class ConvNeXtDecoder(nn.Module):
    def __init__(
        self,
        block_setting,
        remove_layer = 0,
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.remove_layer = remove_layer
        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = InverseCNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        layers.append(
            tg.models.Repe
        )


        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
  

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class InverseCNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            tg.models.Permute([0, 3, 1, 2]),
            nn.Linear(in_features= dim, out_features=4 *dim, bias=True),
            nn.GELU(),
            tg.models.Permute([0, 2, 3, 1]),
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            norm_layer(dim),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = ops.StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result