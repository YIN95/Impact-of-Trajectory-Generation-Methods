import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, dim, activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
            
        # self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2)

        self.fc1 = nn.Linear(dim, 16)
        self.fc2 = nn.Linear(16, 4)

        self.logic = nn.Linear(4, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        # x, _ = self.lstm(x, None)
        # x = self.activation(self.fc1(x[0, -1, :]))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        prob = torch.sigmoid(self.logic(x))
        return prob