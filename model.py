import torch
import torch.nn as nn

class PricePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(PricePredictionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.dropout(x)
        x = self.relu2(self.layer2(x))
        x = self.output_layer(x)
        return x