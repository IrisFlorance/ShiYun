import torch
from torch import nn


class LatentProjector(nn.Module):
    def __init__(self, encoder_width, config):
        super().__init__()
        width = config["width"]
        if width % 4:
            raise ValueError("Projector width must be divisible by four.")
        self.input = nn.Sequential(nn.LayerNorm(encoder_width), nn.Linear(encoder_width, width))
        self.queries = nn.Parameter(torch.randn(config["tokens"], width) * 0.15)
        self.pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, config["dim"]))

    def forward(self, hidden, mask):
        features = self.input(hidden.float())
        queries = self.queries.unsqueeze(0).expand(len(hidden), -1, -1)
        pooled, _ = self.pool(
            queries, features, features, key_padding_mask=~mask, need_weights=False
        )
        return self.output(pooled + queries).float()
