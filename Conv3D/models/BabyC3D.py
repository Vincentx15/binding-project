import torch.nn as nn


class Crazy(nn.Module):
    def __init__(self):
        super(Crazy, self).__init__()
        import pickle
        import torch
        ligand_path = 'data/ligands/whole_dict_embed_128.p'
        ligands_dict = pickle.load(open(ligand_path, 'rb'))
        import numpy as np
        lig_emb = np.array(list(ligands_dict.values()))
        avg = np.mean(lig_emb, axis=0)[:, np.newaxis].T
        print(avg.shape)
        self.avg = torch.Tensor(avg).cuda()
        print(self.avg.size())
        self.group1 = nn.Linear(128, 128)


    '''
    (1, 128)
    torch.Size([1, 128])
    '''

    def forward(self, x):
        x = x.view(1, -1)[:, :128]
        # print('x', x.size())
        # print('avg', self.avg.size())
        out = self.avg + 0.00000001 * self.group1(x)
        # print('out', out.size())
        return out


'''
first     torch.Size([128, 512])
second torch.Size([128, 128])
'''

# C3D Model
class BabyC3D(nn.Module):
    def __init__(self):
        super(BabyC3D, self).__init__()
        self.group1 = nn.Sequential(
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(4, 32, kernel_size=4, padding=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm3d(32))

        self.group2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm3d(64))

        self.group3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm3d(256)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),  #
            nn.ReLU(),
            nn.Dropout(0.3))

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128))  # 101

        self._features = nn.Sequential(
            self.group1,
            self.group2,
            self.group3
        )

        self._classifier = nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, x):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return out
