"""C model: a single linear layer mapping [z_t, h_t] to an action (paper Eq. 1).

    a_t = W_c [z_t ; h_t] + b_c

With z=32 and h=256 this has (32+256)*3 + 3 = 867 parameters - small enough to
be optimised by CMA-ES instead of back-propagation.
"""

import numpy as np

from .config import ACTION_SIZE, RNN_HIDDEN, Z_SIZE


def controller_input_size(mode: str) -> int:
    return Z_SIZE + RNN_HIDDEN if mode == "zh" else Z_SIZE


class Controller:
    def __init__(self, mode: str = "zh"):
        assert mode in ("zh", "z"), "mode 'zh' = full World Model, 'z' = V-model-only ablation"
        self.mode = mode
        self.input_size = controller_input_size(mode)
        self.num_params = self.input_size * ACTION_SIZE + ACTION_SIZE
        self.set_params(np.zeros(self.num_params))

    def set_params(self, params):
        params = np.asarray(params, dtype=np.float32)
        n = self.input_size * ACTION_SIZE
        self.W = params[:n].reshape(self.input_size, ACTION_SIZE)
        self.b = params[n:]

    def act(self, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        x = np.concatenate([z, h]) if self.mode == "zh" else z
        a = np.tanh(x @ self.W + self.b)
        # Map the tanh outputs onto CarRacing's action ranges (as in the paper's code):
        a[1] = (a[1] + 1.0) / 2.0          # gas   in [0, 1]
        a[2] = np.clip(a[2], 0.0, 1.0)     # brake in [0, 1]
        return a
