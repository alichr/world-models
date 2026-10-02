"""Dataset utilities: rollout files on disk -> batches for the V and M models."""

from pathlib import Path

import numpy as np


def list_rollouts(data_dir):
    files = sorted(Path(data_dir).glob("rollout_*.npz"))
    if not files:
        raise FileNotFoundError(f"No rollout_*.npz files in {data_dir}. Run scripts/01_collect_data.py first.")
    return files


def split_files(files, val_fraction=0.05):
    n_val = max(1, int(len(files) * val_fraction))
    return files[:-n_val], files[-n_val:]


def iterate_frame_batches(files, batch_size, rng, files_per_chunk=50):
    """Stream shuffled uint8 frame batches (B,64,64,3) without loading everything.

    Files are shuffled, read ``files_per_chunk`` at a time into RAM, and frames
    inside that chunk are shuffled. This scales from 100 to 10,000 rollouts.
    """
    order = rng.permutation(len(files))
    for start in range(0, len(order), files_per_chunk):
        chunk = np.concatenate([np.load(files[i])["obs"] for i in order[start:start + files_per_chunk]])
        perm = rng.permutation(len(chunk))
        for b in range(0, len(perm) - batch_size + 1, batch_size):
            yield chunk[perm[b:b + batch_size]]


class SeriesDataset:
    """Pre-encoded episodes (mu, logvar, actions) used to train the MDN-RNN."""

    def __init__(self, path, episode_ids=None):
        # Read each array once: every d["key"] access on an NpzFile re-reads it from disk.
        with np.load(path) as d:
            mu, logvar, actions, lengths = d["mu"], d["logvar"], d["actions"], d["lengths"]
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        ids = range(len(lengths)) if episode_ids is None else episode_ids
        self.episodes = [
            (mu[offsets[i]:offsets[i + 1]], logvar[offsets[i]:offsets[i + 1]], actions[offsets[i]:offsets[i + 1]])
            for i in ids
        ]

    def sample_batch(self, batch_size, seq_len, rng):
        """Random windows of length seq_len+1, with z re-sampled from N(mu, sigma)."""
        eligible = [e for e in self.episodes if len(e[0]) > seq_len]
        if not eligible:
            raise ValueError(f"No episode longer than seq_len={seq_len}; lower --seq-len.")
        z, a = [], []
        for idx in rng.integers(len(eligible), size=batch_size):
            mu, logvar, act = eligible[idx]
            s = rng.integers(0, len(mu) - seq_len)
            m, lv = mu[s:s + seq_len + 1], logvar[s:s + seq_len + 1]
            z.append(m + np.exp(0.5 * lv) * rng.standard_normal(m.shape, dtype=np.float32))
            a.append(act[s:s + seq_len])
        z = np.stack(z)
        return z[:, :-1], np.stack(a), z[:, 1:]  # inputs z_t, a_t ; target z_{t+1}
