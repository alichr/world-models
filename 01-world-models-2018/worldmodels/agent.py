"""The full World Model agent (V + M + C) acting in the real CarRacing env."""

from pathlib import Path

import numpy as np
import torch

from .controller import Controller
from .env import preprocess, to_tensor_frames
from .mdrnn import MDNRNN
from .vae import VAE


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vae(path, device="cpu") -> VAE:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    vae = VAE(ckpt["z_size"]).to(device)
    vae.load_state_dict(ckpt["model"])
    return vae.eval()


def load_mdrnn(path, device="cpu") -> MDNRNN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    rnn = MDNRNN(hidden=ckpt["hidden"], n_mix=ckpt["n_mix"]).to(device)
    rnn.load_state_dict(ckpt["model"])
    return rnn.eval()


class WorldModelAgent:
    """Runs on CPU, one environment step at a time."""

    def __init__(self, vae_path, mdrnn_path, mode="zh"):
        self.vae = load_vae(vae_path)
        self.rnn = load_mdrnn(mdrnn_path) if mode == "zh" else None
        self.controller = Controller(mode)
        self.reset()

    @classmethod
    def from_models(cls, vae, rnn, controller):
        """Build an agent from model objects (e.g. to mix trained and untrained parts)."""
        agent = cls.__new__(cls)
        agent.vae, agent.rnn, agent.controller = vae.eval(), rnn.eval() if rnn is not None else None, controller
        agent.reset()
        return agent

    def reset(self):
        self.state = self.rnn.initial_state() if self.rnn is not None else None

    @torch.no_grad()
    def encode(self, frame64: np.ndarray):
        mu, logvar = self.vae.encode(to_tensor_frames(frame64[None]))
        # The paper samples z (rather than using mu) when acting.
        return VAE.reparameterize(mu, logvar)[0]

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        z = self.encode(preprocess(obs))
        h = self.state[0][0, 0].numpy() if self.rnn is not None else None
        action = self.controller.act(z.numpy(), h)
        if self.rnn is not None:
            # h_{t+1} = RNN(h_t, z_t, a_t)
            self.state = self.rnn.step(z, torch.from_numpy(action), self.state)
        return action


def run_episode(env, agent, seed=None, max_steps=1000, early_stop=0, on_step=None):
    """Play one episode and return the total reward.

    early_stop > 0 ends the episode after that many consecutive steps without
    visiting a new track tile (a laptop speed-up; the paper always plays 1000).
    """
    obs, _ = env.reset(seed=seed)
    agent.reset()
    total, since_tile = 0.0, 0
    for _ in range(max_steps):
        action = agent.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if on_step is not None:
            on_step(obs, action, reward)
        since_tile = 0 if reward > 0 else since_tile + 1
        if terminated or truncated or (early_stop and since_tile >= early_stop):
            break
    return total


def checkpoint_paths(ckpt_dir):
    d = Path(ckpt_dir)
    return d / "vae.pt", d / "mdrnn.pt"
