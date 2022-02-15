import torch
from torch import nn
from torch.nn import functional as F
from .non_local import NONLocalBlock2D
from .utils import CALayer, BasicConv2d
from .decoder import Decoder, FocalFuseAttention, FocalGuidedRefine
from .convgru import ConvGRUCell

class FuseNet(nn.Module):
    def __init__(self):
        super(FuseNet, self).__init__()

        self.non_local = NONLocalBlock2D(in_channels=64, inter_channels=16, sub_sample=False, bn_layer=True)

        self.focal_fuse0 = FocalFuseAttention(in_channels=64)
        self.focal_fuse1 = FocalFuseAttention(in_channels=64)
        self.focal_fuse2 = FocalFuseAttention(in_channels=64)
        self.focal_fuse3 = FocalFuseAttention(in_channels=64)
        self.focal_fuse4 = FocalFuseAttention(in_channels=64)

        self.gru = ConvGRUCell(input_channels=64, hidden_channels=64, kernel_size=3)
        self.tran = nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1)

        self.refine = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
                                    nn.ReLU())

        self.pred0 = nn.Conv2d(in_channels=64, out_channels=1, stride=1, kernel_size=3, padding=1)
        self.pred1 = nn.Conv2d(in_channels=64, out_channels=1, stride=1, kernel_size=3, padding=1)
        self.pred2 = nn.Conv2d(in_channels=64, out_channels=1, stride=1, kernel_size=3, padding=1)
        self.pred3 = nn.Conv2d(in_channels=64, out_channels=1, stride=1, kernel_size=3, padding=1)
        self.pred4 = nn.Conv2d(in_channels=64, out_channels=1, stride=1, kernel_size=3, padding=1)


    def forward(self, rgb_feature, focal_feature, lowlevel_feature):
        focal_feature_0 = self.non_local(focal_feature, rgb_feature)
        x0 = self.focal_fuse0(focal_feature_0)
        output0 = self.gru(x0, rgb_feature)
        pred0 = self.pred0(F.upsample(output0, scale_factor=4, mode="bilinear"))

        focal_feature_1 = self.non_local(focal_feature_0, output0)
        x1 = self.focal_fuse1(focal_feature_1)
        output1 = self.gru(x1, output0)
        pred1 = self.pred1(F.upsample(output1, scale_factor=4, mode="bilinear"))

        focal_feature_2 = self.non_local(focal_feature_1, output1)
        x2 = self.focal_fuse2(focal_feature_2)
        output2 = self.gru(x2, output1)
        pred2 = self.pred2(F.upsample(output2, scale_factor=4, mode="bilinear"))

        focal_feature_3 = self.non_local(focal_feature_2, output2)
        x3 = self.focal_fuse3(focal_feature_3)
        output3 = self.gru(x3, output2)
        pred3 = self.pred3(F.upsample(output3, scale_factor=4, mode="bilinear"))

        focal_feature_4 = self.non_local(focal_feature_3, output3)
        x4 = self.focal_fuse4(focal_feature_4)
        output4 = self.gru(x4, output3)

        output4 = F.upsample(output4, scale_factor=2, mode="bilinear")
        output4 = self.refine(output4 + self.tran(lowlevel_feature))

        pred4 = self.pred4(F.upsample(output4, scale_factor=2, mode="bilinear"))


        return pred0, pred1, pred2, pred3, pred4
