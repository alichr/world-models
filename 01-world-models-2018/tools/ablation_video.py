"""Which part of the World Model makes the car drive? A 2x2 video on one track.

    1. V untrained,  M untrained,  C untrained
    2. V trained,    M untrained,  C untrained
    3. V trained,    M untrained,  C trained
    4. V trained,    M trained,    C trained    (the full World Model)

"Untrained" = freshly initialised network weights. The untrained controller
uses small random weights (like a first CMA-ES sample) instead of zeros, so it
actually reacts to what V and M give it. The trained controller in 3 and 4 is
the same one (trained together with the trained M).

    python tools/ablation_video.py --controller runs/snapshots/gen30.npy --out runs/snapshots/ablation.mp4
"""

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from worldmodels.agent import WorldModelAgent, checkpoint_paths, load_mdrnn, load_vae, run_episode
from worldmodels.controller import Controller
from worldmodels.env import make_env
from worldmodels.mdrnn import MDNRNN
from worldmodels.vae import VAE

TRACK_SEED = 20_000_000  # first track of 06_evaluate.py, same as the earlier videos


def build_scenarios(ckpt_dir, controller_path, seed):
    torch.manual_seed(seed)
    vae_raw, rnn_raw = VAE(), MDNRNN()
    vae_path, rnn_path = checkpoint_paths(ckpt_dir)
    vae, rnn = load_vae(vae_path), load_mdrnn(rnn_path)

    def controller(params):
        c = Controller("zh")
        c.set_params(params)
        return c

    random_params = np.random.default_rng(seed).normal(0, 0.1, Controller("zh").num_params)
    trained_params = np.load(controller_path)
    return [
        ("1. V untrained | M untrained | C untrained", WorldModelAgent.from_models(vae_raw, rnn_raw, controller(random_params))),
        ("2. V trained | M untrained | C untrained", WorldModelAgent.from_models(vae, rnn_raw, controller(random_params))),
        ("3. V trained | M untrained | C trained", WorldModelAgent.from_models(vae, rnn_raw, controller(trained_params))),
        ("4. V trained | M trained | C trained (full)", WorldModelAgent.from_models(vae, rnn, controller(trained_params))),
    ]


def record(agent, seed):
    env = make_env(render_mode="rgb_array")
    frames, scores = [], []

    def on_step(obs, action, reward):
        frames.append(env.render())
        scores.append((scores[-1] if scores else 0.0) + reward)

    run_episode(env, agent, seed=seed, on_step=on_step)
    env.close()
    return frames, scores


def tile(frame, title, score, step, done):
    bar = Image.new("RGB", (frame.shape[1], 34), (30, 90, 40) if "full" in title else (40, 40, 60))
    status = f"score {score:7.1f}" + ("   [episode over]" if done else f"   step {step}")
    draw = ImageDraw.Draw(bar)
    draw.text((10, 4), title, fill=(255, 255, 255))
    draw.text((10, 19), status, fill=(255, 230, 120))
    return np.concatenate([np.asarray(bar), frame])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--controller", default="runs/controller_zh/best.npy")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--track", type=int, default=TRACK_SEED)
    p.add_argument("--seed", type=int, default=0, help="seed for the untrained weights")
    p.add_argument("--out", default="runs/ablation.mp4")
    args = p.parse_args()
    torch.set_num_threads(1)

    runs = []
    for title, agent in build_scenarios(args.checkpoints, args.controller, args.seed):
        frames, scores = record(agent, args.track)
        print(f"{title:45s} score {scores[-1]:7.1f}  ({len(frames)} steps)", flush=True)
        runs.append((title, frames, scores))

    writer = imageio.get_writer(args.out, fps=50, quality=8, macro_block_size=8)
    gap_v = np.full((434, 8, 3), 255, np.uint8)
    for t in range(max(len(f) for _, f, _ in runs)):
        tiles = []
        for title, frames, scores in runs:
            i = min(t, len(frames) - 1)  # freeze on the last frame once an episode is over
            tiles.append(tile(frames[i], title, scores[i], t + 1, done=t >= len(frames)))
        top = np.concatenate([tiles[0], gap_v, tiles[1]], 1)
        bottom = np.concatenate([tiles[2], gap_v, tiles[3]], 1)
        writer.append_data(np.concatenate([top, np.full((8, top.shape[1], 3), 255, np.uint8), bottom]))
    writer.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
