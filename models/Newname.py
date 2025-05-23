import torch
import torch.nn as nn
from torch.nn import functional as F
from .uvit import UViT


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim = 512
        self.encoder = UViT(
            img_size=256,
            patch_size=16,
            embed_dim=768,
            latent_dim=self.latent_dim  
        )

        self.feature_decoder = nn.Sequential(
            nn.LayerNorm(768),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(1),  
            nn.Flatten(1),
            nn.Linear(768, 512)
        )

        self.decoder = self._build_decoder(self.latent_dim)

        self.feature_dt = nn.Sequential(
            nn.LayerNorm(768),
            Transpose(1, 2),
            nn.AdaptiveAvgPool1d(1),  
            nn.Flatten(1),
            nn.Linear(768, 512)
        )

        self.discriminator = nn.Sequential(
            self.feature_dt,
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )


    def _build_decoder(self, latent_dim):
        return nn.Sequential(
            self.feature_decoder,
            nn.Linear(latent_dim, 8 * 8 * 512),  # [B, 512] → [B, 8 * 8 * 512=32768]
            nn.Unflatten(1, (512, 8, 8)),  # [B, 512, 8, 8]

            ResUpBlock(512, 256),  # [B,256,16,16]
            ResUpBlock(256, 128),  # [B,128,32,32]
            ResUpBlock(128, 64),  # [B,64,64,64]

            nn.ConvTranspose2d(64, 3, 4, stride=4, padding=0),  # [B,3,256,256]
            nn.Tanh()
        )

    def forward(self, real_img, value_matrix):
        z_real = self.encoder(real_img, value_matrix)
        recon_img = self.decoder(z_real)

        value_matrix_fake = torch.ones_like(value_matrix)

        # z_real_no = self.encoder(real_img, value_matrix).detach()
        # fake_img_no = self.decoder(z_real_no).detach()
        # z_fake_no = self.encoder(fake_img_no, value_matrix_fake).detach()
        #
        # real_pred = self.discriminator(z_real_no)
        # fake_pred = self.discriminator(z_fake_no)

        z_real_yes = self.encoder(real_img, value_matrix)
        fake_img_no = self.decoder(z_real_yes).detach()
        z_fake_yes = self.encoder(fake_img_no, value_matrix_fake)

        real_pred = self.discriminator(z_real_yes)
        fake_pred = self.discriminator(z_fake_yes)

        return recon_img, real_pred, fake_pred