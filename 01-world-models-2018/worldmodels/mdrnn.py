"""M model: Mixture-Density-Network LSTM (paper Section 2.2 / Appendix A.2).

The LSTM receives the current latent z_t and action a_t and models
P(z_{t+1} | a_t, z_t, h_t) as a factorised mixture of 5 Gaussians per latent
dimension. Its hidden state h_t is what the controller uses as "memory".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ACTION_SIZE, N_MIXTURES, RNN_HIDDEN, Z_SIZE

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


class MDNRNN(nn.Module):
    def __init__(self, z_size=Z_SIZE, action_size=ACTION_SIZE, hidden=RNN_HIDDEN, n_mix=N_MIXTURES):
        super().__init__()
        self.z_size, self.hidden, self.n_mix = z_size, hidden, n_mix
        self.lstm = nn.LSTM(z_size + action_size, hidden, batch_first=True)
        # For every latent dim: n_mix mixture logits, n_mix means, n_mix log-stds.
        self.fc = nn.Linear(hidden, z_size * n_mix * 3)

    def forward(self, z, a, state=None):
        """z: (B,T,Z), a: (B,T,A) -> mixture params for z_{t+1}, each (B,T,Z,K)."""
        out, state = self.lstm(torch.cat([z, a], dim=-1), state)
        B, T, _ = out.shape
        p = self.fc(out).view(B, T, self.z_size, 3 * self.n_mix)
        logmix, mean, logstd = p.split(self.n_mix, dim=-1)
        logmix = F.log_softmax(logmix, dim=-1)
        return (logmix, mean, logstd), state

    def initial_state(self, batch=1, device="cpu"):
        zeros = torch.zeros(1, batch, self.hidden, device=device)
        return zeros, zeros.clone()

    @torch.no_grad()
    def step(self, z, a, state):
        """One recurrent step (used while acting in the real env or in a dream)."""
        _, state = self.lstm(torch.cat([z, a], dim=-1).view(1, 1, -1), state)
        return state

    @torch.no_grad()
    def sample(self, logmix, mean, logstd, temperature=1.0):
        """Draw z_{t+1} from the mixture; temperature tau controls uncertainty."""
        logits = logmix / temperature
        k = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        mu = mean.gather(-1, k).squeeze(-1)
        sigma = logstd.gather(-1, k).squeeze(-1).exp()
        return mu + sigma * torch.randn_like(mu) * math.sqrt(temperature)


def mdn_loss(logmix, mean, logstd, target):
    """Negative log-likelihood of target (B,T,Z) under the factorised mixture."""
    y = target.unsqueeze(-1)
    log_normal = -0.5 * ((y - mean) / logstd.exp()).pow(2) - logstd - LOG_SQRT_2PI
    return -torch.logsumexp(logmix + log_normal, dim=-1).mean()
