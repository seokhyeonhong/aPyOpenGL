import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, output_padding=0, last_layer=False):
        super(DeconvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding, output_padding=output_padding),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Identity() if last_layer else nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layers(x)

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # encoder layers
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),     # (120, 69) -> (60, 34)
            ConvBlock(32, 64),    # (60, 34)  -> (30, 17)
            ConvBlock(64, 128),   # (30, 17)  -> (15, 8)
            ConvBlock(128, 256),  # (15, 8)   -> (7, 4)
            ConvBlock(256, 256)   # (7, 4)    -> (3, 2)
        )
        self.encoder.apply(self.init_weights)

        # decoder layers
        self.decoder = nn.Sequential(
            DeconvBlock(256, 256, padding=(0, 1), output_padding=(0, 1)),              # (3, 2)    -> (7, 4)
            DeconvBlock(256, 128, padding=(0, 1), output_padding=(0, 1)),              # (7, 4)    -> (15, 8)
            DeconvBlock(128, 64, padding=(1, 0), output_padding=(1, 0)),               # (15, 8)   -> (30, 17)
            DeconvBlock(64, 32, padding=(1, 1), output_padding=(1, 1)),                # (30, 17)  -> (60, 34)
            DeconvBlock(32, 1, padding=(1, 0), output_padding=(1, 0), last_layer=True) # (60, 34)  -> (120, 69)
        )
        self.decoder.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x