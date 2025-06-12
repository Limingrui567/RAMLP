import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== Residual Block with Dilated Convolution ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dilation=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(in_channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return F.leaky_relu(out)

# ========== Encoder with Skip Connections ==========
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_res1 = ResidualBlock(32)
        self.enc_conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_res2 = ResidualBlock(64)
        self.enc_conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_res3 = ResidualBlock(128)

        self.fc1 = nn.Linear(128 * 10 * 5 * 10, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x1 = F.leaky_relu(self.enc_conv1(x))
        x1 = self.enc_res1(x1)

        x2 = F.leaky_relu(self.enc_conv2(x1))
        x2 = self.enc_res2(x2)

        x3 = F.leaky_relu(self.enc_conv3(x2))
        x3 = self.enc_res3(x3)

        x3 = x3.view(x3.shape[0], -1)
        x = F.leaky_relu(self.fc1(x3))
        x = self.fc2(x)
        return x, (x1, x2) 

# ========== Decoder with Skip Connections ==========
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128 * 10 * 5 * 10)

        self.dec_conv1 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_res1 = ResidualBlock(64)
        self.dec_conv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_res2 = ResidualBlock(32)
        self.dec_conv3 = nn.ConvTranspose3d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, skips):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = x.view(-1, 128, 10, 5, 10)

        x = F.leaky_relu(self.dec_conv1(x))
        x = self.dec_res1(x + skips[1])  

        x = F.leaky_relu(self.dec_conv2(x))
        x = self.dec_res2(x + skips[0])  

        x = self.dec_conv3(x)
        return x

# ========== Autoencoder with L2 Regularization ==========
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_vector, skips = self.encoder(x)
        x = self.decoder(latent_vector, skips)
        return x, latent_vector

# Instantiate and test the model
if __name__ == "__main__":
    model = Autoencoder()
    input_tensor = torch.rand(1, 1, 80, 40, 80)  # Batch size = 1, Channels = 1, Depth/Height/Width = 24
    reconstructed, latent_vector = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Reconstructed shape:", reconstructed.shape)
    print("Latent vector shape:", latent_vector.shape)
