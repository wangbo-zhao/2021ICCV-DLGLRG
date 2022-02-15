import torch
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class decoder_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_module, self).__init__()

        self.skip = BasicConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=1, stride=1)

        self.decoding1 = BasicConv2d(in_planes=2 * out_channels, out_planes=out_channels, kernel_size=3, stride=1,
                                     padding=1)

        self.decoding2 = BasicConv2d(in_planes=out_channels, out_planes=out_channels, kernel_size=3, stride=1,
                                     padding=1)

    def forward(self, enc_fea, dec_fea):
        enc_fea = self.skip(enc_fea)

        if dec_fea.size(2) != enc_fea.size(2):
            dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear', align_corners=True)

        dec_fea = torch.cat([enc_fea, dec_fea], dim=1)

        output = self.decoding1(dec_fea)
        output = self.decoding2(output)

        return output


class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
