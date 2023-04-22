import torch
import torch.nn as nn


class MLP_Classifier(nn.Module):

    def __init__(self,
                    input_dim: int,
                    hidden_layers: list,
                    output_dim: int
                    ):
        """
        Initialization of a multi-layer perceptron.

        Parameters:
            input_dim (int): Size of the input.
            hidden_layers (list): List containing the sizes of the hidden layers. Number of elements determines the number of layers.
            output_dim (int): Size of the output.
        """
        
        super(MLP_Classifier, self).__init__()
        hidden_layers = [input_dim] + hidden_layers
        layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)