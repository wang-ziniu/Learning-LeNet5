import torch
from model import LeNet5  # 导入模型定义
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt  # 用于显示图片

# 加载保存的模型
model = LeNet5().cuda()
model.load_state_dict(torch.load('mnist_lenet5.pth'))  # 加载保存的权重
model.eval()  # 切换到推理模式

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 加载测试数据
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

# 获取第一张测试图片和标签
img, label = test_dataset[3782]  # 获取第一张测试图片和标签

# 显示图片
plt.imshow(img.squeeze(), cmap='gray')  # 去掉 channel 维度并显示为灰度图
plt.title(f"真实标签：{label}")
plt.axis('off')  # 去掉坐标轴
plt.show()

# 推理单张图片
with torch.no_grad():
    img = img.unsqueeze(0).cuda()  # 增加 batch 维度并移动到 GPU
    output = model(img)  # 模型推理
    pred = torch.argmax(output, dim=1)  # 获取预测类别
    print(f"真实标签：{label}, 预测结果：{pred.item()}")