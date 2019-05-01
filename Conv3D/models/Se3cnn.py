import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

# The class GatedBlock inherit from the class torch.nn.Module.
# It contains one convolution, some ReLU and multiplications
from .se3cnn.blocks import GatedBlock


class AvgSpacial(nn.Module):
    def forward(self, inp):  # pylint: disable=W
        # inp [batch, features, x, y, z]
        return inp.view(inp.size(0), inp.size(1), -1).mean(-1)  # [batch, features]


class Se3cnn(nn.Module):

    def __init__(self):
        super().__init__()

        # The parameters of a GatedBlock are:
        # - The representation multiplicities (scalar, vector and dim. 5 repr.) for the input and the output
        # - the non linearities for the scalars and the gates (None for no non-linearity)
        # - stride, padding... same as 2D convolution
        # features = [
        #     (4,),  # As input we have a scalar field
        #     (2, 2, 2, 2),  # Note that this particular choice of multiplicities it completely arbitrary
        #     (4, 4, 3, 3),
        #     (4, 4, 3, 3),
        #     (4, 4, 3, 3),
        #     (4, 4, 3, 3),
        #     (4, 4, 3, 3),
        #     (512,)  # scalar fields to end with fully-connected layers
        # ]

        features = [
            (4,),  # As input we have a scalar field
            (4, 4, 4),
            (8, 8, 4),
            (8, 8, 4, 4),
            (16, 16, 8, 8),
            (16, 16, 8, 8),
            (512,)  # scalar fields to end with fully-connected layers
        ]

        common_block_params = {
            'size': 5,
            'stride': 2,
            'padding': 3,
            'normalization': 'batch',
        }

        block_params = [
            {'activation': (None, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': (F.relu, torch.sigmoid)},
            {'activation': None},
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [GatedBlock(features[i], features[i + 1], **common_block_params, **block_params[i]) for i in
                  range(len(block_params))]

        self.sequence = nn.Sequential(
            *blocks,
            AvgSpacial(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, inp):  # pylint: disable=W
        '''
        :param inp: [batch, features, x, y, z]
        '''

        x = self.sequence(inp)  # [batch, features]

        return x
