import torch.nn as nn
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,6,5) #输入通道1，输出通道6，卷积核5x5
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5) #输入通道6，输出通道15，卷积核5x5
        self.fc1=nn.Linear(16*4*4,120) #输入维度16*4*4，输出维度120
        self.fc2=nn.Linear(120,84) #输入维度120，输出维度84
        self.fc3=nn.Linear(84,10)
    
    def forward(self, x):  
        x = self.pool(nn.functional.relu(self.conv1(x)))  
        x = self.pool(nn.functional.relu(self.conv2(x)))  
        x = x.view(-1, 16*4*4)  # 展平  
        x = nn.functional.relu(self.fc1(x))  
        x = nn.functional.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x  