from model import LeNet5  
import torch  
import torch.optim as optim  
import torch.nn as nn
from torchvision import datasets,transforms
from DatasetDataLoader import train_loader
import wandb
wandb.init(project="mnist-train")  # 初始化wandb

model = LeNet5().cuda()  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  

# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 初始化早停参数
best_loss = float('inf')
patience = 2  # 容忍连续2次loss不下降
counter = 0

for epoch in range(10):  
    running_loss = 0.0  
    for i, (inputs, labels) in enumerate(train_loader):  
        inputs, labels = inputs.cuda(), labels.cuda()  
        optimizer.zero_grad()  # 清空梯度！  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  

        running_loss += loss.item()  
        wandb.log({"loss": loss.item(), "epoch": epoch + 1, "batch": i + 1})  # 记录损失

        if i % 100 == 99:  # 每100个batch打印一次  
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}')  
            running_loss = 0.0

    # 每个 epoch 结束后更新学习率
    scheduler.step()
    # 记录当前学习率到 wandb
    current_lr = scheduler.get_last_lr()[0]
    wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})

    # 计算当前 epoch 的平均损失
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {epoch_loss:.3f}")

    # 早停逻辑
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), 'mnist_lenet5.pth')  # 保存当前最佳模型
        print(f"Epoch {epoch+1}: Loss improved to {best_loss:.3f}, model saved.")
    else:
        counter += 1
        print(f"Epoch {epoch+1}: Loss did not improve. Counter: {counter}/{patience}")
        if counter >= patience:
            print("早停！")
            # 保存最后一次模型
            torch.save(model.state_dict(), 'mnist_lenet5.pth') # 保存当前模型
            print("模型已保存（早停触发）。")
            break