"""U-Net predicting the gain map from the first-day velocity fields."""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with ReLU, optionally followed by 2-D dropout."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net with ``len(features)`` encoder/decoder levels.

    The output is cropped by ``cropped`` cells on each side, so a 16x16 input
    gives a 10x10 gain map with the default ``cropped=3``.

    dropout_mode: "bottleneck" applies ``dropout`` in the bottleneck only,
                  "decoder+bottleneck" also in every decoder block.
    """

    def __init__(self, in_channels=4, out_channels=1, cropped=3,
                 features=(32, 64, 128), dropout=0.0, dropout_mode="bottleneck"):
        super().__init__()
        self.cropped = cropped
        self.features = list(features)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs = nn.ModuleList()
        channels = in_channels
        for feature in self.features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        bottleneck_dropout = dropout if dropout_mode in ("bottleneck", "decoder+bottleneck") else 0.0
        self.bottleneck = DoubleConv(self.features[-1], 2 * self.features[-1],
                                     dropout=bottleneck_dropout)

        decoder_dropout = dropout if dropout_mode == "decoder+bottleneck" else 0.0
        self.ups = nn.ModuleList()
        for feature in reversed(self.features):
            self.ups.append(nn.ConvTranspose2d(2 * feature, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(2 * feature, feature, dropout=decoder_dropout))

        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for level, skip in enumerate(reversed(skips)):
            x = self.ups[2 * level](x)
            _, _, h, w = x.shape
            x = torch.cat((skip[:, :, :h, :w], x), dim=1)
            x = self.ups[2 * level + 1](x)

        x = self.final_conv(x)

        c = self.cropped
        if c > 0:
            x = x[:, :, c:-c, c:-c]
        return x
