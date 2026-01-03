import os
import argparse
import torch
from Src.Models.AttentionUnet import get_attention_unet
from torchvision import transforms
from Src.DataPrepare.cifar10 import get_dataloader_single_class

# 全局参数
DATASET_NAME = 'cifar10'
DATA_ROOT_DIR = "/root/autodl-tmp/Dataset/Cifar10/"
BATCH_SIZE = 64
DEVICE = torch.device("cuda")
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def save_reconstructed_images(input_images, reconstructed_images, output_dir, category_name):
    # 创建类别的子目录
    category_dir = os.path.join(output_dir, category_name)
    os.makedirs(category_dir, exist_ok=True)

    for i, (input_image, reconstructed_image) in enumerate(zip(input_images, reconstructed_images)):
        input_image_path = os.path.join(category_dir, f'{i}_input.png')
        input_image.save(input_image_path)
        reconstructed_image_path = os.path.join(category_dir, f'{i}_reconstructed.png')
        reconstructed_image.save(reconstructed_image_path)


def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='原始类别（如 dog）')
    parser.add_argument('--model_path', type=str, required=True, help='保存的模型权重路径')
    parser.add_argument('--output_dir', type=str, default="gen_cifar", help='输出目录，用于保存图片')
    args = parser.parse_args()

    print(f"使用设备: {DEVICE}")
    print("加载数据集...")

    dataloaders = get_dataloader_single_class(DATA_ROOT_DIR, target_class=args.source, batch_size=BATCH_SIZE)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    model = get_attention_unet().to(DEVICE)

    model = load_model(model, args.model_path, DEVICE)

    # 2. 从dataloader中抽取100个图片，并用模型重建
    input_images = []
    reconstructed_images = []
    for i, (images, _) in enumerate(val_loader):
        if len(input_images) >= 100:  # 修正：判断列表长度更准确
            break
        images = images.to(DEVICE)
        input_images.extend(images.cpu())
        with torch.no_grad():
            reconstructed_images_batch = model(images)
            reconstructed_images.extend(reconstructed_images_batch.detach().cpu())

    # 反归一化逻辑 (针对 CIFAR10 ImageNet 标准) ---
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


    def denormalize(tensor):
        # 截断到 [0, 1] 范围防止溢出
        return (tensor * std + mean).clamp(0, 1)


    # 转换成PIL图片
    transform_to_pil = transforms.ToPILImage()
    input_images = [transform_to_pil(denormalize(image)) for image in input_images[:100]]
    reconstructed_images = [transform_to_pil(denormalize(image)) for image in reconstructed_images[:100]]

    # 3. 创建类别子目录，存放选中的100个图片和重建出的100个图片
    save_reconstructed_images(input_images, reconstructed_images, args.output_dir, args.source)

    print(f"图片已保存到 {args.output_dir}/{args.source}/, 共计 {len(input_images)} 张")