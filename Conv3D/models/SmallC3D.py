import torch.nn as nn



# C3D Model
class SmallC3D(nn.Module):
    def __init__(self):
        super(SmallC3D, self).__init__()
        self.group1 = nn.Sequential(
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            nn.Conv3d(512, 512, kernel_size=2, padding=1, groups=2),
            nn.ReLU(),
            nn.BatchNorm3d(512),
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),  #
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5))

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128))  # 101

        self._features = nn.Sequential(
            self.group1,
            self.group2,
            self.group3,
            self.group4
        )

        self._classifier = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3
        )

    def forward(self, x):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return out
