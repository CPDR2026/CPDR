import torch
import torch.nn as nn
import torchvision.models as models

def get_mobilenet(pretrained, num_classes=10):
    model = models.mobilenet_v2(pretrained=pretrained)  # 使用预训练模型
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


if __name__ == '__main__':
    model = get_mobilenet(pretrained=False, num_classes=10)
    test_input = torch.randn(64, 3, 28, 28)
    output = model(test_input)
    print("Output shape:", output.shape)
