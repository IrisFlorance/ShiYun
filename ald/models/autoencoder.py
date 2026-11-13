import torch
from torch import nn
from torch.nn import functional as F


class LatentAutoencoder(nn.Module):
    def __init__(self, projector, decoder, config):
        super().__init__()
        self.projector = projector
        self.decoder = decoder
        self.options = config["latent"]

    def loss(self, features, attention, ids, targets):
        latent = self.projector(features, attention)
        noisy = latent
        if self.training and self.options["noise"]:
            noisy = latent + self.options["noise"] * torch.randn_like(latent)
        sums, counts = self.decoder.nll(noisy, ids, targets)
        # Estimate spread across examples and query slots so microbatches of one
        # still produce a gradient. Diagnostics separately measure between-text spread.
        spread = (latent.flatten(0, 1).var(0, unbiased=False) + 1e-6).sqrt()
        regularizer = F.relu(self.options["min_std"] - spread).square().mean()
        loss = sums.sum() / counts.sum() + self.options["variance_weight"] * regularizer
        return loss, sums.detach().sum(), counts.detach().sum()
