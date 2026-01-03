import os
import argparse
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from Src.Models.AttentionUnet import get_attention_unet
from Src.DataPrepare.fashionmnist import get_dataloader_single_class

# **全局参数**
DATASET_NAME = 'fashionmnist'
DATA_ROOT_DIR = "/root/autodl-tmp/Dataset/FashionMNIST/"
BATCH_SIZE = 64
DEVICE = torch.device("cuda")

# **FashionMNIST 类别**
FASHIONMNIST_CLASSES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot'
]


def save_reconstructed_images(input_images, reconstructed_images, output_dir, category_name):
    category_dir = os.path.join(output_dir, category_name)
    os.makedirs(category_dir, exist_ok=True)

    for i, (input_image, reconstructed_image) in enumerate(zip(input_images, reconstructed_images)):
        input_image_path = os.path.join(category_dir, f'{i}_input.png')
        input_image.save(input_image_path)

        reconstructed_image_path = os.path.join(category_dir, f'{i}_reconstructed.png')
        reconstructed_image.save(reconstructed_image_path)


def tensor_to_pil(image_tensor):
    """
    由于训练时 Normalize(0.5, 0.5)，此处必须 x * 0.5 + 0.5
    """
    # 1. 反归一化
    img = image_tensor * 0.5 + 0.5
    # 2. 截断防止溢出，并转为 PIL
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)

def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    # **命令行参数**
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='原始类别（如 Dress）')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default="gen_fashionmnist", help='输出目录')
    args = parser.parse_args()

    # **检查类别是否有效**
    if args.source not in FASHIONMNIST_CLASSES:
        raise ValueError(f"类别 '{args.source}' 无效，可选类别: {FASHIONMNIST_CLASSES}")

    print(f"使用设备: {DEVICE}")
    print("加载数据集...")

    dataloaders = get_dataloader_single_class(DATA_ROOT_DIR, target_class=args.source, batch_size=BATCH_SIZE)
    val_loader = dataloaders['val']

    model = get_attention_unet().to(DEVICE)

    model = load_model(model, args.model_path, DEVICE)

    input_list, reconstructed_list = [], []

    for i, (images, _) in enumerate(val_loader):
        if len(input_list) >= 100:
            break
        images = images.to(DEVICE)

        with torch.no_grad():
            outputs = model(images)
            outputs = outputs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # **保存到列表 (注意只取前100张)**
        input_list.extend(images.cpu())
        reconstructed_list.extend(outputs.cpu())

    final_input_images = [tensor_to_pil(img) for img in input_list[:100]]
    final_reconstructed_images = [tensor_to_pil(img) for img in reconstructed_list[:100]]

    save_reconstructed_images(final_input_images, final_reconstructed_images, args.output_dir, args.source)

    print(f"图片已保存至 {args.output_dir}/{args.source}/，共计 {len(final_input_images)} 张。")