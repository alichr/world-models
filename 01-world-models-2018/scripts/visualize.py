"""Look inside the trained models.

    python scripts/visualize.py vae                    # real frames vs VAE reconstructions -> runs/vae_reconstructions.png
    python scripts/visualize.py dream                  # real episode vs MDN-RNN "dream"     -> runs/dream.gif
    python scripts/visualize.py dream --temperature 1.15
    python scripts/visualize.py curves                 # training curves of all stages       -> runs/curves.png
"""

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from worldmodels.agent import checkpoint_paths, load_mdrnn, load_vae
from worldmodels.data import list_rollouts, split_files
from worldmodels.env import to_tensor_frames


def to_uint8(x):
    return (x.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def show_vae(args):
    vae = load_vae(checkpoint_paths(args.checkpoints)[0])
    _, val_files = split_files(list_rollouts(args.data))
    rng = np.random.default_rng(0)
    episodes = [np.load(f)["obs"] for f in rng.choice(val_files, 10)]
    frames = np.stack([obs[rng.integers(len(obs) // 5, len(obs))] for obs in episodes])
    recon = to_uint8(vae.decode(vae.encode(to_tensor_frames(frames))[0]))
    grid = np.concatenate([np.concatenate(frames, 1), np.concatenate(recon, 1)], 0)
    Image.fromarray(grid).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(args.out or "runs/vae_reconstructions.png")
    print("top: real frames, bottom: reconstructions ->", args.out or "runs/vae_reconstructions.png")


@torch.no_grad()
def show_dream(args):
    """Feed the real first frames, then let the MDN-RNN imagine the future.

    The dream is driven by the same actions the real episode used, so the left
    (reality) and right (imagination) halves can be compared frame by frame.
    """
    vae_path, rnn_path = checkpoint_paths(args.checkpoints)
    vae, rnn = load_vae(vae_path), load_mdrnn(rnn_path)
    _, val_files = split_files(list_rollouts(args.data))
    ep = np.load(val_files[args.episode % len(val_files)])
    obs, actions = ep["obs"], torch.from_numpy(ep["actions"])

    state = rnn.initial_state()
    z = vae.encode(to_tensor_frames(obs[:1]))[0]
    frames = []
    for t in range(min(args.steps, len(actions) - 1)):
        (logmix, mean, logstd), state = rnn(z.view(1, 1, -1), actions[t].view(1, 1, -1), state)
        if t < args.warmup:  # condition on reality for the first few frames
            z = vae.encode(to_tensor_frames(obs[t + 1:t + 2]))[0]
        else:
            z = rnn.sample(logmix[0, 0], mean[0, 0], logstd[0, 0], args.temperature).view(1, -1)
        dream = to_uint8(vae.decode(z))[0]
        frames.append(np.concatenate([obs[t + 1], np.full((64, 4, 3), 255, np.uint8), dream], 1))
    out = args.out or "runs/dream.gif"
    imageio.mimsave(out, [np.asarray(Image.fromarray(f).resize((f.shape[1] * 4, 256), Image.NEAREST))
                          for f in frames], duration=1 / 30, loop=0)
    print(f"left: real episode, right: dream (tau={args.temperature}) ->", out)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def show_curves(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    if Path("runs/vae/log.csv").exists():
        rows = read_csv("runs/vae/log.csv")
        ep = [int(r["epoch"]) for r in rows]
        axes[0].plot(ep, [float(r["train_loss"]) for r in rows], "o-", label="train")
        axes[0].plot(ep, [float(r["val_loss"]) for r in rows], "o-", label="val")
        axes[0].set(title="V: VAE loss (recon + KL)", xlabel="epoch")
        axes[0].legend()
    if Path("runs/mdrnn/log.csv").exists():
        rows = read_csv("runs/mdrnn/log.csv")
        st = [int(r["step"]) for r in rows]
        axes[1].plot(st, [float(r["train_nll"]) for r in rows], label="train")
        axes[1].plot(st, [float(r["val_nll"]) for r in rows], label="val")
        axes[1].set(title="M: MDN-RNN negative log-likelihood", xlabel="step")
        axes[1].legend()
    for mode, color in (("zh", "C0"), ("z", "C1")):
        d = Path(f"runs/controller_{mode}")
        if (d / "log.csv").exists():
            rows = read_csv(d / "log.csv")
            g = [int(r["generation"]) for r in rows]
            axes[2].plot(g, [float(r["mean"]) for r in rows], color=color, alpha=0.4, label=f"{mode}: population mean")
            axes[2].plot(g, [float(r["max"]) for r in rows], color=color, lw=0.8, ls=":", label=f"{mode}: population max")
        if (d / "eval.csv").exists():
            rows = read_csv(d / "eval.csv")
            axes[2].errorbar([int(r["generation"]) for r in rows], [float(r["eval_mean"]) for r in rows],
                             [float(r["eval_std"]) for r in rows], color=color, marker="o", capsize=3,
                             label=f"{mode}: eval on unseen tracks")
    axes[2].axhline(900, color="gray", ls="--", lw=0.8)
    axes[2].text(0, 905, "solved (900)", color="gray", fontsize=8)
    axes[2].set(title="C: CMA-ES controller reward", xlabel="generation")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    out = args.out or "runs/curves.png"
    fig.savefig(out, dpi=130)
    print("->", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["vae", "dream", "curves"])
    p.add_argument("--data", default="data/rollouts")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=60, help="real frames fed before dreaming starts")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out")
    args = p.parse_args()
    Path("runs").mkdir(exist_ok=True)
    {"vae": show_vae, "dream": show_dream, "curves": show_curves}[args.what](args)


if __name__ == "__main__":
    main()
