from torch import nn

class CNN1D(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=0, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*349, out_features=512),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

