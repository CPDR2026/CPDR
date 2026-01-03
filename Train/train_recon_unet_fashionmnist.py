import os
import random
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from torch import nn
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from Src.Models.AttentionUnet import get_attention_unet
from Src.DataPrepare.fashionmnist import get_dataloader_single_class

# **全局参数**
DATASET_NAME = 'fashionmnist'
DATA_ROOT_DIR = "/root/autodl-tmp/Dataset/FashionMNIST/"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = f"./ExpResults/recover_{DATASET_NAME}"
FEATURES_DIR = "./fashionmnist_features"
EVAL_INTERVAL = 10  # 每 10 轮评估一次特征偏移
FASHIONMNIST_CLASSES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']
model_path = "/root/autodl-tmp/CPDR-FL/Weights/efficientnetb0_fashionmnist.pth"


# 定义 EfficientNet-B0 分类模型
class EfficientNetB0Classifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetB0Classifier, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.efficientnet.classifier[1].in_features  # 获取最后一层全连接层的输入维度
        self.efficientnet.classifier[1] = nn.Linear(in_features, num_classes)  # 替换全连接层

    def forward(self, x):
        return self.efficientnet(x)


class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super(FeatureExtractor, self).__init__()
        self.model = models.efficientnet_b0(pretrained=False)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 10)
        state_dict = torch.load(model_path, map_location=DEVICE)
        new_state_dict = {k.replace("efficientnet.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=True)
        self.features = self.model.features
        self.avgpool = self.model.avgpool

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        # 放大特征，防止模长过小
        feature = feature / (torch.norm(feature, dim=1, keepdim=True) + 1e-10)
        feature = feature * 20.0
        return feature


def load_pretrained_efficientnet():
    model_path = "/root/autodl-tmp/CPDR-FL/Weights/efficientnetb0_fashionmnist.pth"
    classify_model = EfficientNetB0Classifier().to(DEVICE)
    classify_model.load_state_dict(torch.load(model_path))
    classify_model.eval().to(DEVICE)
    return classify_model


# **读取 FashionMNIST 预存特征**
def load_feature(class_name):
    feature_path = os.path.join(FEATURES_DIR, f"{class_name}.pt")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"特征文件 {feature_path} 不存在，请检查 `features/` 目录")
    return torch.load(feature_path).to(DEVICE)


# **U-Net 适配 FashionMNIST（单通道）**
def get_unet_fashionmnist():
    model = get_attention_unet(in_channels=3, out_channels=3)  # 修改输入/输出通道
    return model.to(DEVICE)


# 图像重建可视化函数
def visualize_reconstruction(model, dataloader, device, save_path=None):
    """可视化模型的图像重建效果"""
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            noisy_images = images.to(device)
            reconstructed_images = model(noisy_images).cpu()
            reconstructed_images = reconstructed_images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # 转为灰度
            noisy_images = noisy_images.cpu().numpy().transpose(0, 2, 3, 1)
            reconstructed_images = reconstructed_images.numpy().transpose(0, 2, 3, 1)

            indices = random.sample(range(len(noisy_images)), 5)
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            for i, idx in enumerate(indices):
                axs[0, i].imshow((noisy_images[idx] - noisy_images[idx].min()) /
                                 (noisy_images[idx].max() - noisy_images[idx].min()))
                axs[1, i].imshow((reconstructed_images[idx] - reconstructed_images[idx].min()) /
                                 (reconstructed_images[idx].max() - reconstructed_images[idx].min()))
                axs[0, i].set_title("Input")
                axs[1, i].set_title("Reconstructed")

            for ax in axs.flatten():
                ax.axis('off')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
            plt.show()
            break


def get_live_avg_feature(dataloader, extractor, class_name):
    """实时计算锚特征：确保 Reference 特征由当前的 extractor 生成"""
    extractor.eval()
    all_feats = []
    print(f"正在实时计算 [{class_name}] 类的锚特征...")
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc=f"校准 {class_name}"):
            images = images.to(DEVICE)
            feat = extractor(images)
            all_feats.append(feat.cpu())
    avg_feat = torch.cat(all_feats, dim=0).mean(dim=0, keepdim=True)
    # 打印模长，方便调试（正常量级应在 10^0 ~ 10^1 之间，若为 10^-14 则有问题）
    print(f"[{class_name}] 校准完成。特征模长: {avg_feat.norm().item():.4f}")
    return avg_feat.to(DEVICE)


def combined_loss(unet_input, unet_output, target_avg_feature, feature_extractor, original_feature, lambda_recon=1.0,
                  lambda_triplet=1.0, margin=1.0):
    reconstruction_loss = F.mse_loss(unet_output, unet_input)
    feature_gen = feature_extractor(unet_output)
    feature_pos = target_avg_feature.repeat(unet_output.size(0), 1)
    feature_neg = original_feature.repeat(unet_output.size(0), 1)

    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)
    triplet_loss = triplet_loss_fn(feature_gen, feature_pos, feature_neg)

    total_loss = lambda_recon * reconstruction_loss + lambda_triplet * triplet_loss
    return total_loss, reconstruction_loss, triplet_loss


# def combined_loss_fl_defense(
#         unet_input, unet_output, target_avg, feature_extractor, original_avg,
#         fed_global_model, benign_direction, target_label,
#         lambda_recon=1.0, lambda_triplet=1.0, lambda_update=1.0, margin=2.0
# ):
#     """
#     fed_global_model: 当前 FL 轮次的全局模型 (用于计算攻击产生的更新方向)
#     benign_direction: 预先计算好的正常更新向量 (Flattened)
#     target_label: 毒化样本的原始标签
#     """
#
#     # 1. 重建损失 (MSE)
#     recon_loss = F.mse_loss(unet_output, unet_input)
#
#     # 2. 投毒效果损失 (特征空间偏移程度)
#     gen_feat = feature_extractor(unet_output)
#     pos_feat = target_avg.expand(gen_feat.size(0), -1)
#     neg_feat = original_avg.expand(gen_feat.size(0), -1)
#     triplet_loss = F.triplet_margin_loss(gen_feat, pos_feat, neg_feat, margin=margin)
#
#     # 3. 模型更新一致性损失
#     # A. 取参数，通常只取最后几层参数
#     # params = [p for p in fed_global_model.parameters() if p.requires_grad]
#     params = [p for name, p in fed_global_model.named_parameters() if 'fc' in name]
#
#     # B. 前向传播：计算毒化样本在“全局模型”下的分类损失
#     logits = fed_global_model(unet_output)
#     l_fed = F.cross_entropy(logits, target_label)
#
#     # C. 计算梯度 dw = d(L_fed) / dw
#     # create_graph=True 使攻击梯度本身对 U-Net 参数也可导
#     grads = torch.autograd.grad(
#         l_fed, params, create_graph=True, retain_graph=True
#     )
#
#     # D. 展平并拼接梯度向量，形成攻击更新方向
#     atk_direction = torch.cat([g.view(-1) for g in grads])
#     atk_direction = -atk_direction
#
#     # E. 计算余弦相似度损失
#     # 希望 atk_direction 靠近 benign_direction，即相似度越大越好，因此 Loss = 1 - Similarity
#     cos_sim = F.cosine_similarity(atk_direction.unsqueeze(0), benign_direction.unsqueeze(0))
#     update_loss = 1 - cos_sim.mean()
#
#     # 综合损失
#     total_loss = (lambda_recon * recon_loss +
#                   lambda_triplet * triplet_loss +
#                   lambda_update * update_loss)
#
#     return total_loss, lambda_recon * recon_loss, lambda_triplet * triplet_loss, lambda_update * update_loss


def evaluate_feature_shift(model, val_loader, classifier, device, target_index):
    model.eval()
    classifier.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 适配 FashionMNIST 灰度图
    ])

    total_images = 0
    shifted_images = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            reconstructed_images = model(images).to(device)

            # resized_images = torch.stack([transform(to_pil_image(img.squeeze(0))) for img in reconstructed_images]).to(device)

            outputs = classifier(reconstructed_images)
            predictions = outputs.argmax(dim=1)

            shifted_images += (predictions == target_index).sum().item()
            total_images += labels.size(0)

    shift_ratio = shifted_images / total_images
    print(
        f"特征偏移率：{shift_ratio:.4%}（{shifted_images}/{total_images} 张图片被分类为 {FASHIONMNIST_CLASSES[target_index]}）")
    return shift_ratio


def train(model, train_loader, val_loader, target_avg_feature, original_feat, feature_extractor, classifier, args):
    print(f"开始训练: {args.source} -> {args.target}")
    target_idx = FASHIONMNIST_CLASSES.index(args.target)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_metrics = {'total': 0.0, 'recon': 0.0, 'triplet': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images, _ in pbar:
            images = images.to(DEVICE)
            outputs = model(images)

            outputs = outputs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            loss, r_loss, t_loss = combined_loss(
                images, outputs, target_avg_feat,
                feature_extractor, original_feat,
                lambda_recon=float(args.lambda_recon),
                lambda_triplet=float(args.lambda_triplet),
                margin=float(args.margin)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_metrics['total'] += loss.item()
            running_metrics['recon'] += r_loss.item()
            running_metrics['triplet'] += t_loss.item()

            pbar.set_postfix({
                'T': f"{loss.item():.9f}",
                'R': f"{r_loss.item():.9f}",
                'Tri': f"{t_loss.item():.9f}"
            })

        avg_total = running_metrics['total'] / len(train_loader)
        scheduler.step(avg_total)

        if (epoch + 1) % EVAL_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            evaluate_feature_shift(model, train_loader, classifier, DEVICE, target_idx)
            save_path = os.path.join(SAVE_DIR, f"epoch{epoch + 1}_recon_{args.source}.png")
            visualize_reconstruction(model, val_loader, DEVICE, save_path)


# **主函数**
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='原始类别')
    parser.add_argument('--target', type=str, required=True, help='目标类别')
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='重建损失系数')
    parser.add_argument('--lambda_triplet', type=float, default=1.0, help='投毒效果损失系数')
    parser.add_argument('--margin', type=float, default=1.0, help='投毒效果损失超参数')
    args = parser.parse_args()

    print("初始化模型中...")
    unet = get_unet_fashionmnist().to(DEVICE)
    ext = FeatureExtractor(model_path).to(DEVICE)
    ext.eval()
    cls = load_pretrained_efficientnet()

    print("准备实时锚特征...")
    source_dataloaders = get_dataloader_single_class(DATA_ROOT_DIR, target_class=args.source, batch_size=BATCH_SIZE)
    target_dataloaders = get_dataloader_single_class(DATA_ROOT_DIR, target_class=args.target, batch_size=BATCH_SIZE)
    target_avg_feat = get_live_avg_feature(target_dataloaders['train'], ext, args.target)
    original_feat = get_live_avg_feature(source_dataloaders['train'], ext, args.source)

    # 开始训练
    train(unet, source_dataloaders['train'], source_dataloaders['val'], target_avg_feat, original_feat, ext, cls, args)
    os.makedirs("./Weights", exist_ok=True)
    model_save_path = f"./Weights/recon_{DATASET_NAME}_{args.source}_{args.target}.pth"
    torch.save(unet.state_dict(), model_save_path)
    print(f"模型已保存为 {model_save_path}")


