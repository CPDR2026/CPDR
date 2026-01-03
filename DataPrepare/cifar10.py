import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset


# CIFAR-10 类别名称映射
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

def get_dataloader_single(data_root_dir, batch_size=32):
    """
    获取完整的 CIFAR-10 训练集和验证集的 DataLoader。

    Args:
        data_root_dir (str): 数据集的根目录。
        batch_size (int): 批量大小。

    Returns:
        dict: {'train': train_loader, 'val': val_loader}
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 统一调整图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root=data_root_dir, train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root=data_root_dir, train=False, transform=transform, download=True)

    # 创建 DataLoader
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }


def get_dataloader_single_class(data_root_dir, target_class, batch_size=32):
    """
    获取 CIFAR-10 数据集中某个特定类别的样本，并返回对应的 DataLoader。

    Args:
        data_root_dir (str): 数据集的根目录。
        target_class (str): 目标类别名称（如 "cat", "dog" 等）。
        batch_size (int): 批量大小。

    Returns:
        dict: {'train': train_loader, 'val': val_loader}
    """
    if target_class not in CIFAR10_CLASSES:
        raise ValueError(f"Invalid target_class: {target_class}. Must be one of {CIFAR10_CLASSES}")
    
    # 获取目标类别索引
    target_class_index = CIFAR10_CLASSES.index(target_class)

    # 加载完整数据集
    dataloaders = get_dataloader_single(data_root_dir, batch_size=batch_size)
    train_dataset, val_dataset = dataloaders['train'].dataset, dataloaders['val'].dataset

    # 过滤出目标类别的样本
    train_subset = Subset(train_dataset, [i for i, label in enumerate(train_dataset.targets) if label == target_class_index])
    val_subset = Subset(val_dataset, [i for i, label in enumerate(val_dataset.targets) if label == target_class_index])

    return {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    }



# 测试函数
if __name__ == "__main__":
    data_root_dir = "~/autodl-tmp/Dataset/Cifar10"
    
    # 获取完整数据集
    dataloaders = get_dataloader_single(data_root_dir, batch_size=64)
    for images, labels in dataloaders['train']:
        print(f"训练集批次图像大小: {images.shape}, 标签大小: {labels.shape}")
        break
    for images, labels in dataloaders['val']:
        print(f"验证集批次图像大小: {images.shape}, 标签大小: {labels.shape}")
        break

    # 获取单个类别数据集
    dataloaders = get_dataloader_single_class(data_root_dir, target_class='dog', batch_size=64)
    for images, labels in dataloaders['train']:
        print(f"'dog' 类训练集批次图像大小: {images.shape}, 标签大小: {labels.shape}")
        break
    for images, labels in dataloaders['val']:
        print(f"'dog' 类验证集批次图像大小: {images.shape}, 标签大小: {labels.shape}")
        break


"""
tensor([0.1960, 0.1597, 0.1799,  ..., 0.1724, 0.1951, 0.1978], device='cuda:0')
tensor([0.1958, 0.1599, 0.1798,  ..., 0.1726, 0.1957, 0.1985], device='cuda:0')
"""