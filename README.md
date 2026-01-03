# 基于数据重建的深度学习投毒攻击

1. 使用torchvision下载数据集到Dataset/，当前支持CIFAR10，FashionMNIST
2. 训练特征提取器
3. 训练重建模型
4. 批量重建数据
5. 手动混合毒化数据与正常数据，在常规训练框架下投毒

（参考 Train/train.sh）

