import torch.nn as nn


# C3D Model
class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(4, 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=4, stride=4))
        self.clf = nn.Linear(1280, 128)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.clf(out)
