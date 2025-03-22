# Learning-LeNet5

## 项目概述
本项目基于 PyTorch 实现了经典的 LeNet-5 模型，用于 MNIST 数据集的手写数字分类任务。以下是学习过程中涉及的主要知识点和代码总结。

## 模型定义
以下是 LeNet-5 模型的定义代码：

```python
import torch.nn as nn

class LeNet5(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入通道1，输出通道6，卷积核5x5
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 最大池化
        self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道6，输出通道16，卷积核5x5
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 全连接层，输入维度16*4*4，输出120
        self.fc2 = nn.Linear(120, 84)  # 全连接层，输入120，输出84
        self.fc3 = nn.Linear(84, 10)  # 全连接层，输入84，输出10
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # 展平
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## 数据预处理
以下是 MNIST 数据集的预处理代码：

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 使用 MNIST 的均值和标准差进行归一化
])

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)
```
## 模型训练
以下是训练代码的核心部分：

```python
import torch.optim as optim
import torch.nn as nn

model = LeNet5().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_loss = float('inf')
patience = 2
counter = 0

for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), 'mnist_lenet5.pth')
    else:
        counter += 1
        if counter >= patience:
            print("早停触发，训练结束。")
            break
```

## 模型推理
以下是推理代码的核心部分：

```python
import torch
from torchvision import datasets, transforms

model = LeNet5().cuda()
model.load_state_dict(torch.load('mnist_lenet5.pth'))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
img, label = test_dataset[0]

with torch.no_grad():
    img = img.unsqueeze(0).cuda()
    output = model(img)
    pred = torch.argmax(output, dim=1)
    print(f"真实标签：{label}, 预测结果：{pred.item()}")
```
## 可视化测试图片
以下是显示测试图片的代码：

```python
import matplotlib.pyplot as plt

plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"真实标签：{label}")
plt.axis('off')
plt.show()
```
## 总结
学习内容：
1. 使用 PyTorch 定义经典的 LeNet-5 模型。
2. 数据预处理的重要性及其实现方法。
3. 模型训练的完整流程，包括损失函数、优化器、学习率调度器和早停机制。
4. 模型推理的实现及测试图片的可视化。
5. 张量操作与自动求导的基本用法。
6. 如何使用 wandb 进行实验记录和可视化。
