import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

# Utils Modules 
# Here I simply adjusted the implementation of the model presented in the last session.

class NormLayer(nn.Module):
    """ Layer that computer embedding normalization """
    def __init__(self, l=2):
        """ Layer initializer """
        assert l in [1, 2]
        super().__init__()
        self.l = l
        return
    
    def forward(self, x):
        """ Normalizing embeddings x. The shape of x is (B,D) """
        x_normalized = x / torch.norm(x, p=self.l, dim=-1, keepdim=True)
        return x_normalized

# An adapted version of the Siamese model that uses a ResNet-18 backbone that can be either pretrained or not.
# We remove the classifier layers from the backbone.

class SiameseModel(nn.Module):
    """ 
    Implementation of a simple siamese model 
    """
    def __init__(self, emb_dim=32, non_linearity=None, pretrained=True):
        """ Module initializer """
        super().__init__()
        
        # convolutional feature extractor
        resnet = tv.models.resnet18(weights='DEFAULT' if pretrained else None)
        # Remove classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # fully connected embedder
        self.fc = []
        self.fc.append(nn.Linear(512, emb_dim))
        if non_linearity is not None:
            if non_linearity == 'relu':
                self.fc.append(nn.ReLU())
            elif non_linearity =='sigmoid':
                self.fc.append(nn.Sigmoid())
            elif non_linearity == 'tanh':
                self.fc.append(nn.Tanh())
            else:
                raise NotImplementedError
        self.fc = nn.Sequential(*self.fc)
        
        # auxiliar layers
        self.flatten = nn.Flatten()
        self.norm = NormLayer()
    
        return
    
    def forward(self, x):
        """ Forwarding a triplet """
        x = self.backbone(x)
        x_flat = self.flatten(x)
        x_emb = self.fc(x_flat)
        x_emb_norm = self.norm(x_emb)
        return x_emb_norm