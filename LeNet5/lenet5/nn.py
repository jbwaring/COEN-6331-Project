from rich import print
import torch
import torch.nn as nn


class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes, path=None, use_big_hidden_layer=False):
        super(ConvNeuralNet, self).__init__()
        self.USE_BIG_HIDDEN_LAYER = use_big_hidden_layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        if (self.USE_BIG_HIDDEN_LAYER):
            self.fc = nn.Linear(400, 2048)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(2048, 84)
        else:
            self.fc = nn.Linear(400, 120)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        if (path):
            try:
                self.load(path)
            except:
                print("Error loading model from path")
        self.HDCMode = False

    def enableHDC(self):
        self.HDCMode = True

    def disableHDC(self):
        self.HDCMode = False

    def forward(self, x):
        if (self.HDCMode is True):
            return self.forwardWithoutLastLayer(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def forwardWithoutLastLayer(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        if (self.USE_BIG_HIDDEN_LAYER):
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = torch.sign(out)
            out = (out + 1) / 2  # map -1 to 0 and 1 to 1

        out = out.reshape(out.size(0), -1)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
