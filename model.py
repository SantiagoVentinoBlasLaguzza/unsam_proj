import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BetaVAE(nn.Module):
    def __init__(self, latent_dim=250, hidden_dim=1024*5, beta=0.1, dropout_rate=0.01, input_channels=3):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels

        # Encoder layers
        self.enc_conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc_bn4 = nn.BatchNorm2d(256)

        self.flatten_size = 256 * 7 * 7

        # FC layers for mu and logvar
        self.fc1 = nn.Linear(self.flatten_size, self.hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder FC
        self.fc_decode = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc_bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.fc_decode2 = nn.Linear(self.hidden_dim, self.flatten_size)

        # Decoder transpose conv layers
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_bn3 = nn.BatchNorm2d(32)
        self.dec_conv4 = nn.ConvTranspose2d(32, input_channels, 4, 2, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.enc_bn1(self.enc_conv1(x)))
        x = torch.tanh(self.enc_bn2(self.enc_conv2(x)))
        x = torch.tanh(self.enc_bn3(self.enc_conv3(x)))
        x = torch.tanh(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = torch.tanh(self.fc_bn2(self.fc_decode(z)))
        x = torch.tanh(self.fc_decode2(x))
        x = x.view(z.size(0), 256, 7, 7)
        x = torch.tanh(self.dec_bn1(self.dec_conv1(x)))
        x = torch.tanh(self.dec_bn2(self.dec_conv2(x)))
        x = torch.tanh(self.dec_bn3(self.dec_conv3(x)))
        x = torch.sigmoid(self.dec_conv4(x))

        x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        target_size = (x.size(-2), x.size(-1))
        recon_x = self.decode(z, target_size=target_size)
        return recon_x, mu, logvar, z