import torch.nn as nn
import torch


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Fully connected neural network with two hidden layers
class NeuralNet(nn.Module):
    def __init__(self, input_size=88, hidden_size=500, num_classes=3):
        super(NeuralNet, self).__init__()
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, num_classes)
        )


    def forward(self, x):
        out = self.classifier(x)
        return out

class NeuralNetBatchNorm(nn.Module):
    def __init__(self, input_size=88, hidden_size=500, num_classes=3):
        super(NeuralNetBatchNorm, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, num_classes)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out

class NeuralNetDropOut(nn.Module):
    def __init__(self, input_size=88, hidden_size=500, num_classes=3):
        super(NeuralNetDropOut, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_size, 600),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(300, num_classes)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out