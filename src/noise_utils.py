import torch
from torch import nn


class AWGNGenerator(nn.Module):
    def __init__(self, snr):
        super(AWGNGenerator, self).__init__()
        self.noise_cumulative_sigma = 0
        self.snr_ratio = 10 ** (snr / 20)
        self.counter = 0

    def forward(self, proj):
        noise = torch.randn_like(proj)
        ratio = torch.sqrt((proj ** 2).sum()) / torch.sqrt((noise ** 2).sum())
        sigma = ratio / self.snr_ratio
        self.noise_cumulative_sigma = (self.counter * self.noise_cumulative_sigma + sigma) / (self.counter + 1)
        self.counter += 1
        return proj + (noise * self.noise_cumulative_sigma)
