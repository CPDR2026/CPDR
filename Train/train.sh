# 训练cifar的特征提取器
# python train_resnet50_classify_model_cifar10.py

# 训练fashionmnist的特征提取器
# python train_efficient_classify_model_fashionmnist.py

# 训练对cifar的重建模型，例如：
# python Src/Train/train_recon_unet_cifar10.py --source airplane --target ship 2>&1 | tee recon_airplane_ship.log
# python Src/Train/train_recon_unet_cifar10.py --source automobile --target truck 2>&1 | tee recon_automobile_truck.log


# 训练对fashionmnist的重建模型
# python Src/Train/train_recon_unet_fashionmnist.py --source T-shirt --target Shirt 2>&1 | tee recon_T-shirt_Shirt.log
# python Src/Train/train_recon_unet_fashionmnist.py --source Trouser --target Dress 2>&1 | tee recon_Trouser_Dress.log


# 使用对cifar的重建模型生成样例数据
# python Src/Utils/gen_cifar.py --source airplane --model_path Weights/recon_cifar10_airplane_ship.pth
# python Src/Utils/gen_cifar.py --source automobile --model_path Weights/recon_cifar10_automobile_truck.pth


# 使用对fashionmnist的重建模型生成样例数据
# python Src/Utils/gen_fashionmnist.py --source Pullover --model_path Weights/recon_fashionmnist_Pullover_Coat.pth
# python Src/Utils/gen_fashionmnist.py --source Shirt --model_path Weights/recon_fashionmnist_Shirt_T-shirt.pth