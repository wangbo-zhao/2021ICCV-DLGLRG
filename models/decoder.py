import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import BasicConv2d, CALayer
from .non_local import NONLocalBlock2D
from .convgru import ConvGRUCell

class FocalFuseAttention(nn.Module):
    def __init__(self, in_channels):
        super(FocalFuseAttention, self).__init__()
        self.channel_fuse = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.se = CALayer(channel=in_channels, reduction=in_channels // 8)

    def forward(self, input):
        fuse = self.channel_fuse(input)
        fuse_attention = F.softmax(fuse, dim=0)
        output = input * fuse_attention
        output = torch.sum(output, dim=0).unsqueeze(0)

        return output



class decoder_module(nn.Module):
    def __init__(self, in_channels, out_channels, nl=False, refine=False):
        super(decoder_module, self).__init__()

        self.rgb_skip = BasicConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=1, stride=1)
        self.focal_skip = BasicConv2d(in_planes=in_channels, out_planes=out_channels, kernel_size=1, stride=1)

        self.focal_fuse = FocalFuseAttention(in_channels=out_channels)

        if refine is True:

            self.rgb_refine = FocalGuidedRefine()
        else:
            self.rgb_refine = None

        if nl is True:
            self.nl = NONLocalBlock2D(in_channels=out_channels, inter_channels=out_channels // 4, sub_sample=False, bn_layer=True)
        else:
            self.nl = None

        self.decoding1 = BasicConv2d(in_planes=3 * out_channels, out_planes=out_channels, kernel_size=3, stride=1,
                                     padding=1)

        self.decoding2 = BasicConv2d(in_planes=out_channels, out_planes=out_channels, kernel_size=3, stride=1,
                                     padding=1)

    def forward(self, rgb_enc_fea, focal_enc_fea, dec_fea):

        if self.rgb_refine is not None:

            rgb_enc_fea = self.rgb_skip(rgb_enc_fea)
            focal_enc_fea = self.rgb_skip(focal_enc_fea)

            refine_rgb_feature, sim_matrix = self.rgb_refine(rgb_enc_fea, focal_enc_fea)

            if self.nl is not None:
                focal_enc_fea = self.nl(focal_enc_fea)
            focal_enc_fea = self.focal_fuse(focal_enc_fea)

            if dec_fea.size(2) != rgb_enc_fea.size(2):
                dec_fea = F.upsample(dec_fea, size=[rgb_enc_fea.size(2), rgb_enc_fea.size(3)], mode='bilinear', align_corners=True)

            dec_fea = torch.cat([refine_rgb_feature, focal_enc_fea, dec_fea], dim=1)

            output = self.decoding1(dec_fea)
            output = self.decoding2(output)

            return output, sim_matrix


        else:

            rgb_enc_fea = self.rgb_skip(rgb_enc_fea)
            focal_enc_fea = self.rgb_skip(focal_enc_fea)


            if self.nl is not None:
                focal_enc_fea = self.nl(focal_enc_fea)
            focal_enc_fea = self.focal_fuse(focal_enc_fea)

            if dec_fea.size(2) != rgb_enc_fea.size(2):
                dec_fea = F.upsample(dec_fea, size=[rgb_enc_fea.size(2), rgb_enc_fea.size(3)], mode='bilinear',
                                     align_corners=True)

            dec_fea = torch.cat([rgb_enc_fea, focal_enc_fea, dec_fea], dim=1)

            output = self.decoding1(dec_fea)
            output = self.decoding2(output)

            return output


class FocalGuidedRefine(nn.Module):
    def __init__(self):
        super(FocalGuidedRefine, self).__init__()
        self.gru = ConvGRUCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.fuse = BasicConv2d(in_planes=128, out_planes=64, kernel_size=3, stride=1, padding=1)

        self.interchannel = 8
        self.g = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=self.interchannel, kernel_size=1, stride=1),
                               nn.ReLU(),
                               nn.Conv2d(in_channels=self.interchannel, out_channels=self.interchannel, kernel_size=1, stride=1))


    def forward(self, rgb_feature, focal_features):
        focal_name = focal_features.shape[0]

        hidden = None
        for i in range(focal_name):
            focal_fea = focal_features[i:i+1, :, :, :]
            rgb_focal_fuse = self.fuse(torch.cat((focal_fea, rgb_feature), dim=1))

            hidden = self.gru(rgb_focal_fuse, hidden)


        refine_rgb = hidden
        refine_rgb = self.g(refine_rgb)
        refine_rgb = refine_rgb.view(1, self.interchannel, -1)

        sim_matrix = torch.matmul(refine_rgb.permute(0, 2, 1), refine_rgb)

        return hidden, sim_matrix










class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.predict5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.decoder4 = decoder_module(512, 64, nl=True, refine=False)
        self.predict4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.decoder3 = decoder_module(256, 64, nl=True, refine=False)
        self.predict3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.decoder2 = decoder_module(128, 64, nl=True, refine=True)
        self.predict2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)


    def forward(self, afterfuse, rgb_for_decoder, focal_for_decoder):

        rgb_encoder_conv4, rgb_encoder_conv3, rgb_encoder_conv2, rgb_encoder_conv1 = rgb_for_decoder
        focal_encoder_conv4, focal_encoder_conv3, focal_encoder_conv2, focal_encoder_conv1 = focal_for_decoder

        pred5 = self.predict5(afterfuse)
        pred5 = F.upsample(pred5, scale_factor=8, mode='bilinear', align_corners=True)

        dec_4 = self.decoder4(rgb_encoder_conv4, focal_encoder_conv4, afterfuse)
        pred4 = self.predict4(dec_4)
        pred4 = F.upsample(pred4, scale_factor=8, mode='bilinear', align_corners=True)

        dec_3 = self.decoder3(rgb_encoder_conv3, focal_encoder_conv3, dec_4)
        pred3 = self.predict3(dec_3)
        pred3 = F.upsample(pred3, scale_factor=8, mode='bilinear', align_corners=True)

        dec_2, sim = self.decoder2(rgb_encoder_conv2, focal_encoder_conv2, dec_3)
        pred2 = self.predict2(dec_2)
        pred2 = F.upsample(pred2, scale_factor=4, mode='bilinear', align_corners=True)


        return [pred2, pred3, pred4, pred5], sim