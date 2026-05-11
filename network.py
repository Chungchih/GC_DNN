import torch.nn as nn
from layers import QConv2d, QLinear

class MNIST_CNN(nn.Module):
    def __init__(self, Qmax=127, noise_std_lsb=0.0):
        super().__init__()
        self.conv1 = QConv2d(1, 8, 3, padding=1, Qmax=Qmax, noise_std_lsb=noise_std_lsb)
        self.conv2 = QConv2d(8, 16, 3, padding=1, Qmax=Qmax, noise_std_lsb=noise_std_lsb)
        self.fc = QLinear(16*7*7, 10, Qmax=Qmax, noise_std_lsb=noise_std_lsb)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def set_noise(self, std_lsb):
        for m in self.modules():
            if isinstance(m, (QConv2d, QLinear)):
                m.noise_std_lsb = std_lsb