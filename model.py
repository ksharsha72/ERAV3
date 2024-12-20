import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 12, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(12 * 7 * 7, 34)
        self.fc2 = nn.Linear(34, 10)
        self.dropout = nn.Dropout(0.03)
        self.batch_norm1 = nn.BatchNorm2d(10)
        self.batch_norm2 = nn.BatchNorm2d(12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 12 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
