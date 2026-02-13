import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim=128, output_dim=100):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.net(z)
