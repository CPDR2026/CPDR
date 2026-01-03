import torch
import torch.nn as nn

class MiniVGG(nn.Module):
    def __init__(self, num_classes=10, in_channels=1): 
        super(MiniVGG, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 32 个 3x3 卷积核
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 2x2 池化，降低尺寸
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 个 3x3 卷积核
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 个 3x3 卷积核
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),  # 适配 32x32 的 CIFAR-10 和 28x28 的 FashionMNIST
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 获取 Mini-VGG 模型的方法
def get_minivgg(num_classes=10, in_channels=3):
    model = MiniVGG(num_classes=num_classes, in_channels=in_channels)
    return model


if __name__ == "__main__":
    test_input = torch.randn(64, 3, 28, 28)
    model = get_minivgg(num_classes=10)
    output = model(test_input)
    print("Output shape:", output.shape)
