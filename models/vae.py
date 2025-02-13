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

        # -------------------
        # ENCODER (Convs)
        # -------------------
        self.enc_conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)

        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)

        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(128)

        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.enc_bn4 = nn.BatchNorm2d(256)

        # Flatten after the final conv
        self.flatten_size = 256 * 7 * 7

        # -------------------
        # ENCODER (FC layers)
        # -------------------
        self.fc1 = nn.Linear(self.flatten_size, self.hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.hidden_dim)
        
        # --- ADDED EXTRA LAYER ---
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_bn2_extra = nn.BatchNorm1d(self.hidden_dim)
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # -------------------
        # DECODER (FC layers)
        # -------------------
        self.fc_decode = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc_bn2 = nn.BatchNorm1d(self.hidden_dim)

        self.fc_decode2 = nn.Linear(self.hidden_dim, self.flatten_size)

        # ---------------------------
        # DECODER (Transposed Convs)
        # ---------------------------
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(128)

        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)

        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(32)

        self.dec_conv4 = nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convolutional downsampling
        x = torch.tanh(self.enc_bn1(self.enc_conv1(x)))
        x = torch.tanh(self.enc_bn2(self.enc_conv2(x)))
        x = torch.tanh(self.enc_bn3(self.enc_conv3(x)))
        x = torch.tanh(self.enc_bn4(self.enc_conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # First FC layer
        x = torch.tanh(self.fc_bn1(self.fc1(x)))

        # Extra FC layer (NEW) before mu/logvar
        x = torch.tanh(self.fc_bn2_extra(self.fc2(x)))

        # Dropout (you can place it anywhere that makes sense, here is one option)
        x = self.dropout(x)

        # Get mu and logvar
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        # FC layers to unflatten
        x = torch.tanh(self.fc_bn2(self.fc_decode(z)))
        x = torch.tanh(self.fc_decode2(x))
        x = x.view(z.size(0), 256, 7, 7)

        # Transposed conv upsampling
        x = torch.tanh(self.dec_bn1(self.dec_conv1(x)))
        x = torch.tanh(self.dec_bn2(self.dec_conv2(x)))
        x = torch.tanh(self.dec_bn3(self.dec_conv3(x)))
        x = torch.sigmoid(self.dec_conv4(x))  # final output is usually in [0,1]

        # Optionally resize the final output if needed
        x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        target_size = (x.size(-2), x.size(-1))  # (H, W)
        recon_x = self.decode(z, target_size=target_size)
        return recon_x, mu, logvar, z
