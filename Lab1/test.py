import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载测试数据集
testset = datasets.FashionMNIST(root='../../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 创建改进后的LeNet模型
model = ImprovedLeNet()

# 加载预训练模型参数（如果有）
# model.load_state_dict(torch.load('improved_lenet_model.pth'))

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 测试模型
model.eval()
correct = 0
total = 0
inference_time = 0.0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        start_time = time.time()
        outputs = model(images)
        inference_time += time.time() - start_time
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy * 10000:.4f}%")
print(f"Average Inference Time per Image: {inference_time / total * 1000:.4f} ms")
