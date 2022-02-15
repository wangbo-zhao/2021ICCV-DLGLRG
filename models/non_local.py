import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)



        self.rgb_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.rgb_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.rgb_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        if bn_layer:
            self.rgb_W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )

        else:
            self.rgb_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                                 kernel_size=1, stride=1, padding=0)

    def forward(self, x, rgb):
        '''
        :param x: (N, c, h, w)
        :return:
        '''
        #
        focal_num = x.size(0)
        h, w = x.size(2), x.size(3)
        kernel_size = 3

        #################################################focalfocal#####################################33
        g_x = self.g(x)

        g_x_d1 = F.unfold(g_x, kernel_size=kernel_size, stride=1, dilation=1, padding=1)
        g_x_d1 = g_x_d1.view(focal_num, -1, kernel_size * kernel_size, h * w)
        g_x_d1 = g_x_d1.permute(3, 0, 2, 1)

        g_x_d3 = F.unfold(g_x, kernel_size=kernel_size, stride=1, dilation=3, padding=3)
        g_x_d3 = g_x_d3.view(focal_num, -1, kernel_size * kernel_size, h * w)
        g_x_d3 = g_x_d3.permute(3, 0, 2, 1)
        g_x_d3 = torch.cat((g_x_d3[:, :, :4, :], g_x_d3[:, :, 5:, :]), dim=2)

        g_x = torch.cat((g_x_d1, g_x_d3), dim=2)
        g_x = g_x.reshape(h * w, -1, self.inter_channels)  # [h*w, c, (3*3-1)*12]

        theta_x = self.theta(x)

        theta_x_d1 = F.unfold(theta_x, kernel_size=kernel_size, stride=1, dilation=1, padding=1)
        theta_x_d1 = theta_x_d1.view(focal_num, -1, kernel_size * kernel_size, h * w)
        theta_x_d1 = theta_x_d1.permute(3, 1, 0, 2)

        theta_x_d3 = F.unfold(theta_x, kernel_size=kernel_size, stride=1, dilation=3, padding=3)
        theta_x_d3 = theta_x_d3.view(focal_num, -1, kernel_size * kernel_size, h * w)
        theta_x_d3 = theta_x_d3.permute(3, 1, 0, 2)
        theta_x_d3 = torch.cat((theta_x_d3[:, :, :, :4], theta_x_d3[:, :, :, 5:]), dim=3)

        theta_x = torch.cat((theta_x_d1, theta_x_d3), dim=3)
        theta_x = theta_x.reshape(h * w, self.inter_channels, -1)  # [h*w, c, (3*3)*12]

        phi_x = self.phi(x)
        phi_x = phi_x.view(focal_num, -1, h * w)
        phi_x = phi_x.permute(2, 0, 1)

        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(1, 2, 0).contiguous()
        y = y.view(focal_num, self.inter_channels, h, w)

        W_y = self.W(y)

        #################################################rgb focal#####################################33

        rgb_h, rgb_w = rgb.size(2), rgb.size(3)
        kernel_size = 3

        g_rgb = self.rgb_g(rgb)

        g_rgb_d1 = F.unfold(g_rgb, kernel_size=kernel_size, stride=1, dilation=1, padding=1)
        g_rgb_d1 = g_rgb_d1.view(1, -1, kernel_size * kernel_size, rgb_h * rgb_w)
        g_rgb_d1 = g_rgb_d1.permute(3, 0, 2, 1)

        g_rgb_d3 = F.unfold(g_rgb, kernel_size=kernel_size, stride=1, dilation=3, padding=3)
        g_rgb_d3 = g_rgb_d3.view(1, -1, kernel_size * kernel_size, rgb_h * rgb_w)
        g_rgb_d3 = g_rgb_d3.permute(3, 0, 2, 1)
        g_rgb_d3 = torch.cat((g_rgb_d3[:, :, :4, :], g_rgb_d3[:, :, 5:, :]), dim=2)

        g_rgb = torch.cat((g_rgb_d1, g_rgb_d3), dim=2)
        g_rgb = g_rgb.reshape(rgb_h * rgb_w, -1, self.inter_channels)  # [h*w, c, (3*3-1)*12]

        theta_rgb = self.rgb_theta(rgb)

        theta_rgb_d1 = F.unfold(theta_rgb, kernel_size=kernel_size, stride=1, dilation=1, padding=1)
        theta_rgb_d1 = theta_rgb_d1.view(1, -1, kernel_size * kernel_size, rgb_h * rgb_w)
        theta_rgb_d1 = theta_rgb_d1.permute(3, 1, 0, 2)

        theta_rgb_d3 = F.unfold(theta_rgb, kernel_size=kernel_size, stride=1, dilation=3, padding=3)
        theta_rgb_d3 = theta_rgb_d3.view(1, -1, kernel_size * kernel_size, rgb_h * rgb_w)
        theta_rgb_d3 = theta_rgb_d3.permute(3, 1, 0, 2)
        theta_rgb_d3 = torch.cat((theta_rgb_d3[:, :, :, :4], theta_rgb_d3[:, :, :, 5:]), dim=3)

        theta_rgb = torch.cat((theta_rgb_d1, theta_rgb_d3), dim=3)
        # theta_rgb = theta_rgb.reshape(rgb_h*rgb_w, self.inter_channels, -1)#[h*w, c, (3*3)*12]

        # caculate the simality between focal and rgb
        focal_num = x.size(0)
        phi_rgb = self.rgb_phi(x)  # input x!!!!!!!!!!!!!!!!!
        phi_rgb = phi_rgb.view(focal_num, -1, 1, rgb_h * rgb_w)
        phi_rgb = phi_rgb.permute(3, 1, 0, 2)

        w_ = theta_rgb.size(3)
        h_ = phi_rgb.size(2)
        theta_rgb = theta_rgb.repeat(1, 1, h_, 1)
        phi_rgb = phi_rgb.repeat(1, 1, 1, w_)

        concat_feature = torch.cat([theta_rgb, phi_rgb], dim=1)
        f = self.concat_project(concat_feature)
        f = f.view(f.shape[0], f.shape[2], f.shape[3])

        N = f.size(-1)
        f_div_C_rgb = f / N

        y_rgb = torch.matmul(f_div_C_rgb, g_rgb)
        y_rgb = y_rgb.permute(1, 2, 0).contiguous()
        y_rgb = y_rgb.view(focal_num, self.inter_channels, h, w)

        W_y_rgb = self.rgb_W(y_rgb)

        z = W_y + W_y_rgb + x
        return z


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)
