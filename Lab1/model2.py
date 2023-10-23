import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import time
import os

# Device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 35 # 10
num_classes = 10
batch_size = 100 # 100
learning_rate = 0.001
momentum = 0.9

# 定义LeNet模型
# 卷积 -> 池化 -> 卷积 -> 全连接 -> 全连接
# Batch Normalization 与 MORE ReLu
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                          nn.BatchNorm2d(6),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                          nn.BatchNorm2d(16),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Linear(4 * 4 * 16, 120),
                                       nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84),
                                       nn.ReLU())
        self.fc3 = nn.Linear(84, num_classes)
        
    # 更高效，不需要保留每一层的中间结果
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载数据集
trainset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(os.path.join(script_dir, '../../data'), download=True, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# 创建模型
model = LeNet(num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train_time = 0.0
total_step = len(train_loader)
start = time.time()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item())) 
torch.cuda.synchronize()
train_time = time.time() - start
print(f"Total Train Time: {train_time: .10f} s")

# 测试模型
model.eval()
correct = 0.0
total = 0.0
inference_time = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        start = time.time()
        outputs = model(images)
        torch.cuda.synchronize()
        inference_time += time.time() - start
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print ('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
print(f"Average Inference Time per Image: {inference_time / total: .10f} s")
print(f"Total Inference Time: {inference_time: .10f} s")

# 导出模型参数，也可以自定义导出模型参数的文件格式，这里使用了最简单的方法，但请注意，如果改动了必须保证程序二能够正常读取
for name, param in model.named_parameters():
    np.savetxt(os.path.join(script_dir, f'./{name}.txt'), param.detach().cpu().numpy().flatten())

# Save the model.
torch.save(model.state_dict(), 'LeNet.ckpt')