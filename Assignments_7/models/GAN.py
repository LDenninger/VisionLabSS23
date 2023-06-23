import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Simple convolutional block: Conv + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", None]
        padding = kernel_size // 2
        
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
            
        if dropout is not None:
            block.append(nn.Dropout(dropout))
            
        self.block =  nn.Sequential(*block)

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y


class ConvTransposeBlock(nn.Module):
    """
    Simple convolutional block: ConvTranspose + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Tanh", None]
        padding = kernel_size // 2
        
        block = []
        block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
        if dropout is not None:
            block.append(nn.Dropout(dropout))
            
        self.block =  nn.Sequential(*block)

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y
    


class Generator(nn.Module):
    """
        Convolutional generator module.
    """
    def __init__(self, latent_dim: int, hidden_dims: list(int), kernel_size:int, strides: list(int), num_channels: int=3):
        """ Model initializer """
        super().__init__()

        layers = []
        for i, dim in enumerate(hidden_dims):
            layers.append(
                ConvTransposeBlock(
                        in_channels=latent_dim if i == 0 else hidden_dims[i-1],
                        out_channels=dim,
                        kernel_size=kernel_size,
                        stride=strides[i],
                        add_norm=True,
                        activation="LeakyReLU"
                    )
                )
        layers.append(
            ConvTransposeBlock(
                    in_channels=dim,
                    out_channels=num_channels,
                    kernel_size=4,
                    stride=2,
                    add_norm=False,
                    activation="Tanh"
                )
            )
        
        self.model = nn.Sequential(*layers)
        return
    
    def forward(self, x):
        """ Forward pass through generator """
        y = self.model(x)
        return y



class Discriminator(nn.Module):
    """ A fully convolutional discriminator using LeakyReLU activations. 
    Takes as input either a real or fake sample and predicts its autenticity.
       (B, num_channels, 32, 32)  -->  (B, 1, 1, 1)
    """
    def __init__(self, latent_dim: int, hidden_dims: list(int), kernel_size:int, strides: list(int), dropout: bool, num_channels: int=3):
        """ Module initializer """
        super().__init__()  
        
        layers = []
        for i, dim in enumerate(hidden_dims):
            layers.append(
                ConvBlock(
                        in_channels=num_channels if i == 0 else hidden_dims[i-1],
                        out_channels=dim,
                        kernel_size=kernel_size,
                        add_norm=True,
                        activation="LeakyReLU",
                        dropout=dropout,
                        stride=strides[i]
                    )
                )
        layers.append(
                ConvBlock(
                        in_channels=dim,
                        out_channels=latent_dim,
                        kernel_size=4,
                        stride=4,
                        add_norm=False,
                        activation="Sigmoid"
                    )
                )
        
        self.model = nn.Sequential(*layers)
        return
      
    def forward(self, x):
        """ Forward pass """
        y = self.model(x)
        return y

class GAN(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.generator = Generator(
            latent_dim=self.config['model']['latent_dim'],
            hidden_dims=self.config['model']['hidden_dims'],
            kernel_size=self.config['model']['kernel_size'],
            strides=self.config['model']['strides'],
            num_channels=self.config['model']['input_size'][0]
        ).to(self.device)

        self.discriminator = Discriminator(
            latent_dim=self.config['model']['latent_dim'][::-1],
            hidden_dims=self.config['model']['hidden_dims'],
            kernel_size=self.config['model']['kernel_size'],
            strides=self.config['model']['strides'],
            num_channels=self.config['model']['input_size'][0]
        ).to(self.device)

    def forward(self, x):
        """ Forward pass """
        return self.generator(x)
    
    def generate(self, x):
        return self.generator(x)
    
    def discriminate(self, x):
        return self.discriminator(x)