import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 1000),
            nn.ReLU(),
            nn.Linear(1000, n_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_dim, 1000), 
            nn.Sigmoid(),
            nn.Linear(1000, n_input),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded