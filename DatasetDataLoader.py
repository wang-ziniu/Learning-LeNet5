from torchvision import datasets,transforms
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) #MINST归一化参数
])
train_data=datasets.MNIST (
    root="data",
    train=True,
    download=True,
    transform=transform
)


from torch.utils.data import DataLoader
import os
train_loader =  DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

#测试一个batch
for images,labels in train_loader:
    print(f"图像尺寸：{images.shape}")
    print(f"标签：{labels}")
    break # 只测试一个batch

import matplotlib.pyplot as plt

for images, labels in train_loader:
    # 显示第一张图片
    plt.imshow(images[0].squeeze(), cmap="gray")
    plt.title(f"Label: {labels[0].item()}")
    plt.show()
    break

