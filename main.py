import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt # 导入 matplotlib

# 定义模型保存路径
MODEL_SAVE_DIR = 'output'
PLOT_SAVE_DIR = 'output' # 图表保存路径
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# 定义超参数
BATCH_SIZE = 64
EPOCHS = 10 # 训练轮数，可以根据需要调整以达到 98% 准确率
LEARNING_RATE = 0.001

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(), # 将图片转换为 Tensor
    transforms.Normalize((0.1307,), (0.3081,)) # 归一化处理
])

# 下载或加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 下载或加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. 构建 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1: 输入通道 1, 输出通道 32, 卷积核 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 卷积层2: 输入通道 32, 输出通道 64, 卷积核 3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 最大池化层: 窗口 2x2, 步长 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1: 输入特征 64 * 7 * 7, 输出特征 128
        # MNIST 图片大小 28x28 -> Conv1 -> 28x28 -> Pool1 -> 14x14 -> Conv2 -> 14x14 -> Pool2 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2: 输入特征 128, 输出特征 10 (0-9 十个类别)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # CNN 流程: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> ReLU -> FC
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        # 输出层不使用 ReLU，因为 CrossEntropyLoss 会处理
        x = self.fc2(x)
        return x

model = CNN().to(device)
print("Model Structure:")
print(model)

# 3. 创建优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 4. 训练和评估模型
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]	Loss: {loss.item():.6f}')
    epoch_loss = running_loss / len(train_loader) # 计算 epoch 的平均 loss
    return epoch_loss

def test(): # 移除 epoch 参数，因为不再需要在 test 函数内打印 epoch
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader) # 修改为 test_loader 的长度，得到每个 batch 的平均 loss
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)')
    return test_loss, accuracy # 返回 test loss 和 accuracy

# 5. 绘图函数
def plot_metrics(train_losses, test_losses, test_accuracies):
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 保存图表
    plot_save_path = os.path.join(PLOT_SAVE_DIR, 'training_metrics.png')
    plt.savefig(plot_save_path)
    print(f"Saved training metrics plot to {plot_save_path}")
    # plt.show() # 如果需要在运行时显示图表，取消此行注释


if __name__ == '__main__':
    # 用于存储每个 epoch 的指标
    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []
    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        epoch_train_loss = train(epoch)
        epoch_test_loss, current_accuracy = test() # 调用修改后的 test 函数

        # 记录指标
        train_loss_history.append(epoch_train_loss)
        test_loss_history.append(epoch_test_loss)
        test_accuracy_history.append(current_accuracy)

        # 保存准确率最高的模型
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            # 清理旧的最佳模型文件（可选）
            for f in os.listdir(MODEL_SAVE_DIR):
                if f.startswith('mnist_cnn_best_acc_'):
                    os.remove(os.path.join(MODEL_SAVE_DIR, f))
            model_save_path = os.path.join(MODEL_SAVE_DIR, f'mnist_cnn_best_acc_{best_accuracy:.2f}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path} with accuracy {best_accuracy:.2f}%")

    print(f"Training finished. Best test accuracy achieved: {best_accuracy:.2f}%")
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'mnist_cnn_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

    # 绘制并保存指标曲线
    plot_metrics(train_loss_history, test_loss_history, test_accuracy_history) 