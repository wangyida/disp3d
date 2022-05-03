import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class Periodics(nn.Module):
    def __init__(self,
                 dim_input=2,
                 dim_output=512,
                 is_first=True,
                 transpose=False):
        super(Periodics, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.is_first = is_first
        self.transpose = transpose
        self.with_frequency = True
        self.with_phase = True
        # Omega determines the upper frequencies
        self.omega_0 = 30
        if self.with_frequency:
            if self.with_phase:
                self.Li = nn.Conv1d(
                    self.dim_input, self.dim_output, 1,
                    bias=self.with_phase).cuda()
            else:
                self.Li = nn.Conv1d(
                    self.dim_input,
                    self.dim_output // 2,
                    1,
                    bias=self.with_phase).cuda()
            # nn.init.normal_(B.weight, std=10.0)
            with torch.no_grad():
                if self.is_first:
                    self.Li.weight.uniform_(-1 / self.dim_input,
                                            1 / self.dim_input)
                else:
                    self.Li.weight.uniform_(
                        -np.sqrt(6 / self.dim_input) / self.omega_0,
                        np.sqrt(6 / self.dim_input) / self.omega_0)
        else:
            self.Li = nn.Conv1d(self.dim_input, self.dim_output, 1).cuda()
            self.BN = nn.BatchNorm1d(self.dim_output).cuda()

    def filter(self):
        filters = torch.cat([
            torch.ones(1, self.dim_output // 32 * 32),
            torch.zeros(1, self.dim_output // 32 * 0)
        ], 1).cuda()
        filters = torch.unsqueeze(filters, 2)
        return filters

    def forward(self, x):
        if not torch.is_tensor(x):
