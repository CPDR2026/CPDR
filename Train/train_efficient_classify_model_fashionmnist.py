import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
import os
from tqdm import tqdm

# 全局参数
DATASET_DIR = "/root/autodl-tmp/Dataset/FashionMNIST/"
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 正则化
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "./Weights/efficientnetb0_fashionmnist.pth"
os.makedirs("./Weights", exist_ok=True)

# FashionMNIST 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # EfficientNet 需要更大的输入尺寸
    transforms.Grayscale(num_output_channels=3),  # FashionMNIST 是灰度图，需要转换为 3 通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 适用于 FashionMNIST
])

# 加载数据集
train_dataset = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root=DATASET_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 定义 EfficientNet-B0 分类模型
class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetB0Classifier, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)

        in_features = self.efficientnet.classifier[1].in_features  # 获取最后一层全连接层的输入维度
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)  # 替换全连接层

    def forward(self, x):
        return self.efficientnet(x)

# 初始化模型
model = EfficientNetB0Classifier().to(DEVICE)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 使用 AdamW 作为优化器，提高泛化能力
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)

# 使用 CosineAnnealingLR 作为学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# 训练和测试函数
def train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training") as train_bar:
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_train += predicted.eq(labels).sum().item()
                total_train += labels.size(0)

                train_bar.set_postfix(loss=f"{train_loss / (total_train / BATCH_SIZE):.4f}",
                                      acc=f"{100. * correct_train / total_train:.2f}%")
        train_acc = 100. * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 测试阶段
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            with tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing") as test_bar:
                for inputs, labels in test_bar:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    _, predicted = outputs.max(1)
                    correct_test += predicted.eq(labels).sum().item()
                    total_test += labels.size(0)

                    test_bar.set_postfix(loss=f"{test_loss / (total_test / BATCH_SIZE):.4f}",
                                         acc=f"{100. * correct_test / total_test:.2f}%")
        test_acc = 100. * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # 更新学习率
        scheduler.step()

        # 每 10 轮保存一次模型
        if (epoch + 1) % 10 == 0:
            save_path_epoch = f"./Weights/efficientnetb0_fashionmnist_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), save_path_epoch)
            print(f"模型已保存至 {save_path_epoch}")

    # 最终保存模型
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"最终模型已保存至 {SAVE_PATH}")

# 开始训练和测试
train_and_test(model, train_loader, test_loader, criterion, optimizer, scheduler, NUM_EPOCHS)
