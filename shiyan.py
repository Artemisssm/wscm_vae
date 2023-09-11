import torch.nn as nn
from resnet import *
import torch
from sagan import *
from causal_model import *
import math


class ResMLPBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc2(self.fc1(x))
        out += x
        return self.relu(out)


class bEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, hidden_dims=None):
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 1024, 3, 1, 1),
                nn.BatchNorm2d(1024),
                nn.Conv2d(1024, 2048, 3, 1, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        )
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            ResMLPBlock(1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            ResMLPBlock(256),
            nn.Linear(256, 256)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        avepool = torch.flatten(z, 1)
        z = self.fc(avepool)
        z_mu = self.fc_mu(z)
        z_var = self.fc_var(z)
        return z_mu, z_var


class bDecoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=64, hidden_dims=None, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.decoder_input = nn.Linear(latent_dim, 2048*image_size*image_size)
        modules = []
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                nn.BatchNorm2d(1024),
                nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(512, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ConvTranspose2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ConvTranspose2d(32, 3, 3, 1, 1),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
                nn.Conv2d(3, 3, 3, 1, 1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        x = self.decoder_input(z)
        result = x.view(-1, 2048, self.image_size, self.image_size)
        result = self.decoder(result)
        return result


model = bEncoder()
a = torch.rand(16, 3, 64, 64)
mu, var = model(a)
print(mu, var)
modeld = bDecoder()
re = modeld(mu)
print(re)