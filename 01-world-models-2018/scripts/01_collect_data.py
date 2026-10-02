"""Step 1 - Collect random rollouts of CarRacing (paper: 10,000 rollouts).

Each rollout is saved as data/rollouts/rollout_XXXXX.npz with
    obs      (T, 64, 64, 3) uint8   preprocessed frames o_t
    actions  (T, 3)         float32 a_t taken after seeing o_t
    rewards  (T,)           float32 r_t received after a_t
    dones    (T,)           bool

Resumable: rollouts already on disk are skipped.

    python scripts/01_collect_data.py                      # laptop preset (1,000 rollouts)
    python scripts/01_collect_data.py --preset paper       # 10,000 rollouts
    python scripts/01_collect_data.py --num-rollouts 200   # quick test
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from worldmodels.config import PRESETS, resolve
from worldmodels.env import RandomPolicy, make_env, preprocess


def collect_one(job):
    index, out_dir, max_steps, seed = job
    path = Path(out_dir) / f"rollout_{index:05d}.npz"
    if path.exists():
        return 0
    rng = np.random.default_rng(seed + index)
    env = make_env()
    obs, _ = env.reset(seed=seed + index)
    policy = RandomPolicy(rng)
    frames, actions, rewards, dones = [], [], [], []
    for _ in range(max_steps):
        action = policy()
        frames.append(preprocess(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        if terminated or truncated:
            break
    env.close()
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, obs=np.stack(frames), actions=np.stack(actions).astype(np.float32),
                        rewards=np.array(rewards, np.float32), dones=np.array(dones))
    tmp.rename(path)
    return len(frames)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="laptop", choices=PRESETS)
    p.add_argument("--num-rollouts", dest="num_rollouts", type=int)
    p.add_argument("--max-steps", dest="max_steps", type=int)
    p.add_argument("--out", default="data/rollouts")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = resolve(args, "collect")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.out, cfg["max_steps"], args.seed) for i in range(cfg["num_rollouts"])]
    print(f"Collecting {cfg['num_rollouts']} rollouts with {args.workers} workers -> {args.out}")
    with Pool(args.workers) as pool:
        frames = sum(tqdm(pool.imap_unordered(collect_one, jobs), total=len(jobs), unit="rollout"))
    print(f"Done. {frames:,} new frames written.")


if __name__ == "__main__":
    main()
