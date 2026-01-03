import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """注意力门控模块，用于U-Net的跳跃连接"""
    def __init__(self, in_channels, g_channels, out_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(g_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(out_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        """ x: 跳跃连接的特征图， g: 来自解码器的特征图 """
        g_proj = self.W_g(g)  # 解码器特征图通道对齐
        x_proj = self.W_x(x)  # 跳跃连接特征图通道对齐
        psi = self.relu(g_proj + x_proj)  # 通道相加
        psi = self.sigmoid(self.psi(psi))  # 注意力权重
        return x * psi  # 对跳跃连接加权


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AttentionUNet, self).__init__()

        # 编码器部分
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # 最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈部分
        self.bottleneck = self.double_conv(512, 1024)

        # 解码器部分
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.double_conv(1024, 512)

        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.double_conv(512, 256)

        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.double_conv(256, 128)

        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.double_conv(128, 64)

        # 注意力门控
        self.att4 = AttentionGate(512, 512, 512)
        self.att3 = AttentionGate(256, 256, 256)
        self.att2 = AttentionGate(128, 128, 128)
        self.att1 = AttentionGate(64, 64, 64)

        # 输出卷积层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        """两层卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        """上采样卷积块"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))

        # 解码器路径 + 跳跃连接 + 注意力机制
        g4 = self.upconv4(bottleneck)
        att_enc4 = self.att4(enc4, g4)
        dec4 = torch.cat((g4, att_enc4), dim=1)
        dec4 = self.dec4(dec4)

        g3 = self.upconv3(dec4)
        att_enc3 = self.att3(enc3, g3)
        dec3 = torch.cat((g3, att_enc3), dim=1)
        dec3 = self.dec3(dec3)

        g2 = self.upconv2(dec3)
        att_enc2 = self.att2(enc2, g2)
        dec2 = torch.cat((g2, att_enc2), dim=1)
        dec2 = self.dec2(dec2)

        g1 = self.upconv1(dec2)
        att_enc1 = self.att1(enc1, g1)  # 注意力后的 skip feature
        dec1 = torch.cat((g1, att_enc1), dim=1)
        dec1 = self.dec1(dec1)

        # 输出
        return self.out_conv(dec1)


def get_attention_unet(in_channels=3, out_channels=3):
    model = AttentionUNet(in_channels=in_channels, out_channels=out_channels)  # 输入和输出均为 RGB 图像
    return model


if __name__ == "__main__":
    model = AttentionUNet(in_channels=3, out_channels=3)  # 输入和输出均为 RGB 图像
    x = torch.randn(1, 3, 256, 256)  # 输入示例：一个 256x256 的 RGB 图像
    y = model(x)
    print(y.shape)  # 输出形状应为 (1, 3, 256, 256)
