"""CarRacing environment helpers: creation, frame preprocessing, random policy."""

import gymnasium as gym
import numpy as np
from PIL import Image

from .config import FRAME_SIZE

ENV_ID = "CarRacing-v3"


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)


def preprocess(obs: np.ndarray) -> np.ndarray:
    """96x96x3 game frame -> 64x64x3 uint8, as in the paper.

    The bottom 12 rows (the dashboard with speed / ABS bars) are cropped away
    before resizing, exactly like the reference implementation.
    """
    obs = obs[:84]
    return np.array(Image.fromarray(obs).resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR))


def to_tensor_frames(frames: np.ndarray):
    """(N,64,64,3) uint8 -> (N,3,64,64) float tensor in [0,1]."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(frames))
    return x.permute(0, 3, 1, 2).float().div_(255.0)


class RandomPolicy:
    """Smooth random exploration used to collect the VAE / MDN-RNN dataset.

    The paper collects 10,000 rollouts "from a random policy". Uniformly random
    actions make the car jitter in place (gas and brake cancel out), so, like
    most re-implementations, we use temporally-correlated random actions: the
    steering and throttle drift as a random walk and the brake is tapped only
    occasionally. This makes the car actually drive, drift off the road onto
    grass, and recover - giving the VAE a diverse set of frames.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.action = np.array([0.0, rng.uniform(0.3, 1.0), 0.0], dtype=np.float32)

    def __call__(self) -> np.ndarray:
        a, rng = self.action, self.rng
        a[0] = np.clip(a[0] + rng.normal(0, 0.15), -1.0, 1.0)
        a[1] = np.clip(a[1] + rng.normal(0, 0.10), 0.0, 1.0)
        a[2] = rng.uniform(0.3, 0.8) if rng.random() < 0.05 else 0.0
        return a.copy()
