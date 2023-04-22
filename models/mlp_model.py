import torch
import torch.nn as nn


class MLP_Classifier(nn.Module):

    def __init__(self,
                    input_dim: int,
                    mlp_layers: list,
                    ):
        """
        Initialization of a multi-layer perceptron.

        Parameters:
            input_dim (int): Size of the input.
            mlp_layers (list): List containing the sizes and types of the mlp layers. Number of elements determines the number of layers.
                Format: 
                    [
                        {
                            'type': 'linear',
                            'activation':'relu',
                            'dimension': 128
                        },
                        {
                            'type': 'batchnorm',
                            'eps':1e-5,
                            'momentum': 0.1
                        },
                        {
                            'type': 'dropout',
                            'prob': 0.5
                        }
                    ]
            output_dim (int): Size of the output.
        """
        
        super(MLP_Classifier, self).__init__()
        self.build_up_model(input_dim, mlp_layers)

    def build_up_model(self, input_dim: int, mlp_layers: list):
        layers = nn.ModuleList()
        for (i, layer) in enumerate(mlp_layers):
            layer_inp_dim = input_dim if i == 0 else mlp_layers[i-1]['dimension']
            # Add a linear layer with a given activation function
            if layer['type'] == 'linear':
                layers.append(nn.Linear(layer_inp_dim, layer['dimension']))
                if layer['activation'] is not None:
                    if layer['activation'] =='relu':
                        layers.append(nn.ReLU())
                    if layer['activation'] == 'tanh':
                        layers.append(nn.Tanh())
                    if layer['activation'] =='sigmoid':
                        layers.append(nn.Sigmoid())
                    if layer['activation'] == 'elu':
                        layers.append(nn.ELU())
                    if layer['activation'] =='selu':
                        layers.append(nn.SELU())
                    if layer['activation'] == 'leaky_relu':
                        layers.append(nn.LeakyReLU())
                    if layer['activation'] == 'prelu':
                        layers.append(nn.PReLU())
                    if layer['activation'] == 'softmax':
                        layers.append(nn.Softmax(dim=-1))
            if layer['type'] =='batchnorm':
                layers.append(nn.BatchNorm1d(layer_inp_dim, layer['eps'], layer['momentum']))
            if layer['type'] == 'dropout':
                layers.append(nn.Dropout(layer['prob']))
            
        self.model = nn.Sequential(*layers)
            

    
    def forward(self, x):
        return self.model(x)