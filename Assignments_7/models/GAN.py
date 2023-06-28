import torch
import torch.nn as nn
import torch.nn.functional as F

import torchgadgets as tg



class ConvBlock(nn.Module):
    """
    Simple convolutional block: Conv + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None, downsample=False, add_padding=True):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", None]
        if stride==1:
            padding=kernel_size // 2
        else:
            padding = (kernel_size-stride) // 2
        
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding if add_padding else 0, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
           
        if downsample:
            block.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
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
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None, padding: bool =True):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", None]
  
        
        block = []

        block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=1 if padding else 0, stride=stride))
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
    
class CondLinearEmbedding(nn.Module):
    def __init__(self, emb_size: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.latent_emb = nn.Linear(latent_dim, emb_size)
        self.label_emb = nn.Linear(num_classes, emb_size)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x, y):
        x = self.activation(self.latent_emb(x))
        y = self.activation(self.label_emb(y))
        return x+y


class Generator(nn.Module):
    """
        Convolutional generator module.
    """
    def __init__(self, latent_dim: int, hidden_dims: list, kernel_sizes:list, strides: list, input_size:tuple, embedding: str='linear', conditional: bool=False, num_classes: int = 10):
        """ Model initializer """
        super().__init__()

        layers = []
        emb_size = input_size[0]*input_size[1]*input_size[2] if not conditional else int(input_size[0]*input_size[1]*input_size[2] / 2)
        emb_layers = []
        if embedding=='linear':
            emb_layers.append(nn.Linear(latent_dim, emb_size))
            emb_layers.append(nn.BatchNorm1d(emb_size))
            emb_layers.append(nn.LeakyReLU(negative_slope=0.2))
            emb_layers.append(tg.models.Reshape((-1,input_size[0] if not conditional else int(input_size[0]/2), input_size[1], input_size[2] )))
            self.latent_emb = nn.Sequential(*emb_layers)

            if conditional:
                emb_layers = []
                emb_layers.append(nn.Linear(num_classes, emb_size))
                emb_layers.append(nn.BatchNorm1d(emb_size))
                emb_layers.append(nn.LeakyReLU(negative_slope=0.2))
                emb_layers.append(tg.models.Reshape((-1,input_size[0] if not conditional else int(input_size[0]/2), input_size[1], input_size[2] )))
                self.label_emb = nn.Sequential(*emb_layers)


        elif embedding=='conv':
            emb_layers = []
            emb_layers.append(tg.models.Reshape((-1,latent_dim,1,1)))
            emb_layers.append(
                    ConvTransposeBlock(
                        in_channels=latent_dim,
                        out_channels=input_size[0] if not conditional else int(input_size[0]/2),
                        kernel_size=kernel_sizes[0],
                        stride=strides[0],
                        add_norm=True,
                        padding=False,
                        activation="LeakyReLU",
                    )
                )
            self.latent_emb = nn.Sequential(*emb_layers)
            if conditional:
                emb_layers = []
                emb_layers.append(tg.models.Reshape((-1,num_classes,1,1)))
                emb_layers.append(
                    ConvTransposeBlock(
                        in_channels=num_classes,
                        out_channels=input_size[0] if not conditional else int(input_size[0]/2),
                        kernel_size=kernel_sizes[0],
                        stride=strides[0],
                        add_norm=True,
                        padding=False,
                        activation="LeakyReLU",
                    )
                )
                self.label_emb = nn.Sequential(*emb_layers)

            
        elif embedding=="cond_linear":
            assert conditional, 'Conditional Linear embedding is only available for a conditional GAN'
            self.cond_emb = CondLinearEmbedding(emb_size=input_size[0], latent_dim=latent_dim, num_classes=num_classes)
            self.cond_reshape = tg.models.Reshape((-1,input_size[0],1,1))
        
        for i, dim in enumerate(hidden_dims):
            layers.append(
                ConvTransposeBlock(
                        in_channels=dim,
                        out_channels=hidden_dims[i+1] if i<len(hidden_dims)-1 else 3,
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=True,
                        add_norm=True if i<len(hidden_dims)-1 else False,
                        activation="LeakyReLU" if i<len(hidden_dims)-1 else "Tanh",
                    )
                )
   
        self.model = nn.Sequential(*layers)
        self.conditional = conditional
        self.embedding = embedding
        return
    
    def forward(self, x, y= None):
        """ Forward pass through generator """
        if self.conditional:
            assert y is not None, "Please provide a class label for the conditional GAN"
            if self.embedding=='cond_linear':
                x = self.cond_emb(x, y)
                x = self.cond_reshape(x)
            else:
                x = self.latent_emb(x)
                y = self.label_emb(y)
                x = torch.cat([x,y], dim=1)
        else:
            x = self.latent_emb(x)
            
        out = self.model(x)
        return out



class Discriminator(nn.Module):
    """ A fully convolutional discriminator using LeakyReLU activations. 
    Takes as input either a real or fake sample and predicts its autenticity.
       (B, num_channels, 32, 32)  -->  (B, 1)
    """
    def __init__(self, hidden_dims: list, kernel_sizes:list, strides:list, dropout: bool, num_channels: int=3, add_fc: bool=False, fc_input: int = None, conditional: bool=False, num_classes: int = 10, add_last_padding: bool=True):
        """ Module initializer """
        super().__init__()  
        
        layers = []

        self.img_emb = nn.Sequential(ConvBlock(
                        in_channels=num_channels,
                        out_channels=hidden_dims[0] if not conditional else int(hidden_dims[0]/2),
                        kernel_size=kernel_sizes[0],
                        add_norm=True,
                        stride=strides[0],
                        activation="LeakyReLU",
                        dropout=dropout,
                        downsample=False,
                        add_padding=True
                    )
        )
        if conditional:
            self.label_emb = nn.Sequential(
                ConvBlock(
                        in_channels=num_classes,
                        out_channels=hidden_dims[0] if not conditional else int(hidden_dims[0]/2),
                        kernel_size=kernel_sizes[0],
                        add_norm=True,
                        stride=strides[0],
                        activation="LeakyReLU",
                        dropout=dropout,
                        downsample=False,
                        add_padding=True
                    )
            )
        input_channels = hidden_dims[0]
        hidden_dims = hidden_dims[1:]
        kernel_sizes = kernel_sizes[1:]
        strides = strides[1:]

        for i, dim in enumerate(hidden_dims):
            layers.append(
                ConvBlock(
                        in_channels=input_channels if i == 0 else hidden_dims[i-1],
                        out_channels=dim,
                        kernel_size=kernel_sizes[i],
                        add_norm=True,
                        stride=strides[i],
                        activation="LeakyReLU" if (i<len(hidden_dims)-1 or add_fc) else "Sigmoid",
                        dropout=dropout if (i<len(hidden_dims)-1 or add_fc) else None,
                        downsample=False,
                        add_padding=False if i==i<len(hidden_dims)-1 and not add_last_padding else True
                    )
                )
      
        layers.append(nn.Flatten())
        if add_fc:
            assert fc_input is not None
            layers.append(nn.Linear(fc_input, 1))
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self.conditional = conditional
        return
      
    def forward(self, x, y=None):
        """ Forward pass """
        x = self.img_emb(x)
        if self.conditional:
            assert y is not None, 'Please provide a class label to the conditional GAN'
            y = self.label_emb(y)
            x = torch.cat([x, y], dim=1)
        out = self.model(x)
        return out

class GAN(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        self.config = model_config
        self.conditional = model_config['conditional']
        self.generator = Generator(
            latent_dim=self.config['latent_dim'] ,
            hidden_dims=self.config['generator']['hidden_dims'],
            strides=self.config['generator']['strides'],
            kernel_sizes=self.config['generator']['kernel_sizes'],
            embedding=self.config['generator']['embedding'],
            conditional=model_config['conditional'],
            num_classes=model_config['num_labels'],
            input_size=self.config['generator']['input_size']
        )

        self.discriminator = Discriminator(
            hidden_dims=self.config['discriminator']['hidden_dims'],
            strides=self.config['discriminator']['strides'],
            kernel_sizes=self.config['discriminator']['kernel_sizes'],
            num_channels=self.config['discriminator']['input_size'][0],
            dropout = self.config['discriminator']['dropout'],
            conditional=model_config['conditional'],
            add_fc=self.config['discriminator']['add_fc'],
            num_classes=model_config['num_labels'],
            fc_input=self.config['discriminator']['fc_input'],
            add_last_padding=self.config['discriminator']['add_last_padding']
        )

    def forward(self, x, y=None):
        """ Forward pass """
        return self.generate(x, y)
    
    def generate(self, x, y=None):
        if self.conditional:
            assert y is not None, 'Please provide label for conditional GAN'
        return self.generator(x, y)
    
    def discriminate(self, x, y=None):
        if self.conditional:
            assert y is not None, 'Please provide label for conditional GAN'
            y = torch.repeat_interleave(y.unsqueeze(-1), self.config['discriminator']['input_size'][-2], dim=-1)
            y = torch.repeat_interleave(y.unsqueeze(-1), self.config['discriminator']['input_size'][-1], dim=-1)

        return self.discriminator(x, y)