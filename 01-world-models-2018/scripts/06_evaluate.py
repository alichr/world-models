"""Step 6 - Evaluate a trained World Model agent on unseen tracks; watch or record it.

    python scripts/06_evaluate.py --episodes 100                 # score (paper: 906 +/- 21 over 100 trials)
    python scripts/06_evaluate.py --episodes 3 --render          # watch it drive in a window
    python scripts/06_evaluate.py --episodes 1 --video runs/drive.mp4   # game | VAE input | VAE reconstruction
    python scripts/06_evaluate.py --random                       # baseline: untrained (zero) controller
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from worldmodels.agent import WorldModelAgent, checkpoint_paths, run_episode
from worldmodels.env import make_env, preprocess, to_tensor_frames

EVAL_SEED_OFFSET = 20_000_000  # different from the tracks used during training / CMA-ES eval

_worker = {}


def init_worker(ckpt_dir, mode, params):
    torch.set_num_threads(1)
    agent = WorldModelAgent(*checkpoint_paths(ckpt_dir), mode)
    agent.controller.set_params(params)
    _worker.update(agent=agent, env=make_env())


def play(seed):
    return run_episode(_worker["env"], _worker["agent"], seed=int(seed))


def panel(img, size, label):
    im = Image.fromarray(img).resize((size, size), Image.NEAREST)
    ImageDraw.Draw(im).text((6, 4), label, fill=(255, 255, 255))
    return np.asarray(im)


def record(agent, seed, path, render_mode):
    env = make_env(render_mode="rgb_array" if path else render_mode)
    writer = imageio.get_writer(path, fps=50, quality=8, macro_block_size=8) if path else None

    def on_step(obs, action, reward):
        if writer is None:
            return
        frame = env.render()  # (400, 600, 3)
        small = preprocess(obs)
        with torch.no_grad():
            mu, _ = agent.vae.encode(to_tensor_frames(small[None]))
            recon = (agent.vae.decode(mu)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        side = np.concatenate([panel(small, 200, "VAE input 64x64"), panel(recon, 200, "VAE reconstruction")])
        writer.append_data(np.concatenate([frame, side], axis=1))

    score = run_episode(env, agent, seed=seed, on_step=on_step)
    if writer:
        writer.close()
    env.close()
    return score


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="zh", choices=["zh", "z"])
    p.add_argument("--controller", help="params .npy (default: runs/controller_<mode>/best.npy)")
    p.add_argument("--random", action="store_true", help="use an all-zero controller as a baseline")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--render", action="store_true", help="open a window and watch the agent")
    p.add_argument("--video", help="save an mp4 of the first episode")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    agent = WorldModelAgent(*checkpoint_paths(args.checkpoints), args.mode)
    if args.random:
        params = np.zeros(agent.controller.num_params)
    else:
        params = np.load(args.controller or f"runs/controller_{args.mode}/best.npy")
    agent.controller.set_params(params)
    seeds = EVAL_SEED_OFFSET + np.arange(args.episodes)

    if args.render or args.video:
        if args.video:
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        for i, seed in enumerate(seeds):
            score = record(agent, int(seed), args.video if i == 0 else None, "human" if args.render else None)
            print(f"episode {i + 1}: {score:.1f}", flush=True)
            if args.video and not args.render:
                print(f"saved {args.video}")
                break
        return

    with Pool(args.workers, initializer=init_worker, initargs=(args.checkpoints, args.mode, params)) as pool:
        scores = np.array(pool.map(play, seeds, chunksize=1))
    print(f"{args.mode} controller over {len(scores)} unseen tracks: {scores.mean():.1f} +/- {scores.std():.1f}"
          f"  (min {scores.min():.1f}, max {scores.max():.1f})")


if __name__ == "__main__":
    main()
