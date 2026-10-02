"""Step 3 - Encode every rollout with the trained VAE (frozen).

The MDN-RNN is trained on latent sequences, not pixels. Like the reference
implementation we store mu and log-variance per frame (not a fixed z), so a
fresh z ~ N(mu, sigma) can be drawn every time a sequence is used for training.

    python scripts/03_encode_data.py
Output: data/series.npz with mu, logvar (N_frames, 32), actions (N_frames, 3),
        lengths (N_episodes,)
"""

import argparse

import numpy as np
import torch
from tqdm import tqdm

from worldmodels.agent import get_device, load_vae
from worldmodels.data import list_rollouts
from worldmodels.env import to_tensor_frames


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/rollouts")
    p.add_argument("--vae", default="checkpoints/vae.pt")
    p.add_argument("--out", default="data/series.npz")
    args = p.parse_args()

    device = get_device()
    vae = load_vae(args.vae, device)
    mus, logvars, actions, lengths = [], [], [], []
    for f in tqdm(list_rollouts(args.data), unit="rollout"):
        d = np.load(f)
        mu, logvar = vae.encode(to_tensor_frames(d["obs"]).to(device))
        mus.append(mu.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        actions.append(d["actions"])
        lengths.append(len(d["obs"]))
    np.savez(args.out, mu=np.concatenate(mus), logvar=np.concatenate(logvars),
             actions=np.concatenate(actions), lengths=np.array(lengths))
    print(f"Saved {args.out}: {len(lengths)} episodes, {sum(lengths):,} frames")


if __name__ == "__main__":
    main()
