import torch
import torch.nn as nn

class PricePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(PricePredictionModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.norm1 = nn.LayerNorm(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.layer2 = nn.Linear(128, 64)
        self.norm2 = nn.LayerNorm(64)
        self.relu2 = nn.ReLU()
        
        self.output_layer = nn.Linear(64, 1)
        self.residual = nn.Linear(input_size, 64)

    def forward(self, x):
        identity = self.residual(x)
        
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = x + identity  # Избегаем in-place модификаций
        x = self.output_layer(x)
        return x
