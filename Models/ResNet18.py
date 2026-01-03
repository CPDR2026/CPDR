import torch
import torch.nn as nn
from torchvision import models


def get_resnet18(pretrained, num_classes=10):
    model = models.resnet18(pretrained=pretrained, num_classes=num_classes)
    return model


if __name__ == "__main__":
    # 测试 ResNet-18
    model_resnet = get_resnet18(pretrained=False, num_classes=10)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model_resnet(input_tensor)
    print("ResNet-18 输出尺寸:", output.shape)
