import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

# FashionMNIST 类别名称映射
FASHIONMNIST_CLASSES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot'
]

def get_dataloader_single(data_root_dir, batch_size=32):
    """
    获取完整的 FashionMNIST 训练集和验证集的 DataLoader。

    Args:
        data_root_dir (str): 数据集的根目录。
        batch_size (int): 批量大小。

    Returns:
        dict: {'train': train_loader, 'val': val_loader}
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # EfficientNet 需要更大的输入尺寸
        transforms.Grayscale(num_output_channels=3),  # FashionMNIST 是灰度图，需要转换为 3 通道
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 适用于 FashionMNIST
    ])

    # 加载 FashionMNIST 数据集
    train_dataset = datasets.FashionMNIST(root=data_root_dir, train=True, transform=transform, download=True)
    val_dataset = datasets.FashionMNIST(root=data_root_dir, train=False, transform=transform, download=True)

    # 创建 DataLoader
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }


def get_dataloader_single_class(data_root_dir, target_class, batch_size=32):
    """
    获取 FashionMNIST 数据集中某个特定类别的样本，并返回对应的 DataLoader。

    Args:
        data_root_dir (str): 数据集的根目录。
        target_class (str): 目标类别名称（如 "T-shirt", "Dress" 等）。
        batch_size (int): 批量大小。

    Returns:
        dict: {'train': train_loader, 'val': val_loader}
    """
    if target_class not in FASHIONMNIST_CLASSES:
        raise ValueError(f"Invalid target_class: {target_class}. Must be one of {FASHIONMNIST_CLASSES}")
    
    # 获取目标类别索引
    target_class_index = FASHIONMNIST_CLASSES.index(target_class)

    # 加载完整数据集
    dataloaders = get_dataloader_single(data_root_dir, batch_size=batch_size)
    train_dataset, val_dataset = dataloaders['train'].dataset, dataloaders['val'].dataset

    # 过滤出目标类别的样本
    train_indices = [i for i, label in enumerate(train_dataset.targets) if label == target_class_index]
    val_indices = [i for i, label in enumerate(val_dataset.targets) if label == target_class_index]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    }
