import os
import random
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim, nn
from torchvision import models
from Src.Models.AttentionUnet import get_attention_unet
from Src.DataPrepare.cifar10 import get_dataloader_single_class

# --- 全局参数 ---
DATASET_NAME = 'cifar10'
DATA_ROOT_DIR = "/root/autodl-tmp/Dataset/Cifar10/"
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = f"./ExpResults/recover_{DATASET_NAME}"
EVAL_INTERVAL = 10
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
model_path = "/root/autodl-tmp/CPDR-FL/Weights/resnet50_cifar10.pth"

def normalize_batch(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=10, checkpoint_path=None):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        self.eval()

    def forward(self, x):
        return self.resnet(x)


class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        resnet = models.resnet50(weights=None)
        state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = {k.replace("resnet.", ""): v for k, v in state_dict.items() if "fc." not in k}  # 根据保存方式适配
        resnet.load_state_dict(new_state_dict, strict=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        x = x * 0.5 + 0.5
        # 提取前进行标准化，确保特征模长正确
        x = normalize_batch(x)
        feat = self.features(x)
        return feat.view(feat.size(0), -1)


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
    return avg_feat.to(DEVICE)


def combined_loss(unet_input, unet_output, target_avg, feature_extractor, original_avg, lambda_recon=1.0, lambda_triplet=1.0, margin=2.0):
    # 重建损失 (MSE)
    recon_loss = F.mse_loss(unet_output, unet_input)
    # 特征提取 (内部已包含 normalize)
    gen_feat = feature_extractor(unet_output)
    # 广播特征向量
    pos_feat = target_avg.expand(gen_feat.size(0), -1)
    neg_feat = original_avg.expand(gen_feat.size(0), -1)
    # 投毒效果损失
    triplet_loss_fn = nn.TripletMarginLoss(margin)
    triplet_loss = triplet_loss_fn(gen_feat, pos_feat, neg_feat)

    total_loss = lambda_recon * recon_loss + lambda_triplet * triplet_loss
    return total_loss, lambda_recon * recon_loss, lambda_triplet * triplet_loss


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
    total_images = 0
    shifted_images = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            reconstructed = model(images)

            # 标准化后再输入分类器
            resized = F.interpolate(reconstructed, size=(128, 128), mode='bilinear', align_corners=False)
            normalized = normalize_batch(resized)

            outputs = classifier(normalized)
            predictions = outputs.argmax(dim=1)
            shifted_images += (predictions == target_index).sum().item()
            total_images += labels.size(0)

    shift_ratio = shifted_images / total_images
    print(f"特征偏移率：{shift_ratio:.4%} ({shifted_images}/{total_images})")
    return shift_ratio


def visualize_reconstruction(model, dataloader, device, save_path=None):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            inputs = images.to(device)
            reconstructed = model(inputs).cpu()
            inputs = inputs.cpu()

            def denorm(x):
                return (x - x.min()) / (x.max() - x.min() + 1e-8)

            indices = random.sample(range(len(inputs)), min(5, len(inputs)))
            fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            for i, idx in enumerate(indices):
                axs[0, i].imshow(denorm(inputs[idx]).permute(1, 2, 0))
                axs[1, i].imshow(denorm(reconstructed[idx]).permute(1, 2, 0))
                axs[0, i].axis('off')
                axs[1, i].axis('off')

            plt.tight_layout()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
            plt.close()
            break


# --- 训练循环 ---

def train(model, train_loader, val_loader, target_avg_feat, original_feat, feature_extractor, classifier, args):
    print(f"开始训练: {args.source} -> {args.target}")
    target_idx = CIFAR10_CLASSES.index(args.target)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_metrics = {'total': 0.0, 'recon': 0.0, 'triplet': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for images, _ in pbar:
            images = images.to(DEVICE)
            outputs = model(images)

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
            save_img_path = os.path.join(SAVE_DIR, f"epoch{epoch + 1}_vis.png")
            visualize_reconstruction(model, val_loader, DEVICE, save_img_path)


# --- 主函数 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='原始类别')
    parser.add_argument('--target', type=str, required=True, help='目标类别')
    parser.add_argument('--lambda_recon', type=float, default=1.0, help='重建损失系数')
    parser.add_argument('--lambda_triplet', type=float, default=1.0, help='投毒效果损失系数')
    parser.add_argument('--margin', type=float, default=2.0, help='投毒效果损失超参数')
    args = parser.parse_args()


    print("初始化模型中...")
    unet = get_attention_unet().to(DEVICE)
    ext = FeatureExtractor(model_path).to(DEVICE)
    cls = ResNet50Classifier(num_classes=10, checkpoint_path=model_path).to(DEVICE)

    print("准备实时锚特征...")
    source_dataloaders = get_dataloader_single_class(DATA_ROOT_DIR, target_class=args.source, batch_size=BATCH_SIZE)
    target_dataloaders = get_dataloader_single_class(DATA_ROOT_DIR, target_class=args.target, batch_size=BATCH_SIZE)

    target_avg_feat = get_live_avg_feature(target_dataloaders['train'], ext, args.target)
    original_feat = get_live_avg_feature(source_dataloaders['train'], ext, args.source)

    # 开始训练
    train(unet, source_dataloaders['train'], source_dataloaders['val'], target_avg_feat, original_feat, ext, cls, args)

    os.makedirs("./Weights", exist_ok=True)
    save_name = f"./Weights/recon_{DATASET_NAME}_{args.source}_{args.target}.pth"
    torch.save(unet.state_dict(), save_name)
    print(f"模型已保存至: {save_name}")