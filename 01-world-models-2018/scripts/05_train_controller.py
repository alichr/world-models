"""Step 5 - Train the C model (linear controller) with CMA-ES in the real env.

V and M are frozen. Each generation CMA-ES proposes ``popsize`` parameter
vectors; each is scored by its average reward over ``episodes`` CarRacing
tracks, played in parallel on CPU cores. Every ``eval_every`` generations the
CMA-ES mean is tested on ``eval_episodes`` unseen tracks (the paper reports
this number). Resumable with --resume.

    python scripts/05_train_controller.py                         # laptop preset
    python scripts/05_train_controller.py --mode z                # V-only ablation (no memory)
    python scripts/05_train_controller.py --resume                # continue a stopped run
Outputs in runs/controller_<mode>/: log.csv, eval.csv, best.npy, es.pkl
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import numpy as np
import torch

from worldmodels.agent import WorldModelAgent, checkpoint_paths, run_episode
from worldmodels.config import PRESETS, resolve
from worldmodels.controller import Controller
from worldmodels.env import make_env

EVAL_SEED_OFFSET = 10_000_000  # test tracks never overlap with training tracks

_worker = {}


def init_worker(ckpt_dir, mode, early_stop):
    torch.set_num_threads(1)
    vae_path, rnn_path = checkpoint_paths(ckpt_dir)
    _worker.update(agent=WorldModelAgent(vae_path, rnn_path, mode), env=make_env(), early_stop=early_stop)


def play(job):
    params, seed, early_stop = job
    agent = _worker["agent"]
    agent.controller.set_params(params)
    limit = _worker["early_stop"] if early_stop else 0
    return run_episode(_worker["env"], agent, seed=int(seed), early_stop=limit)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="laptop", choices=PRESETS)
    p.add_argument("--popsize", type=int)
    p.add_argument("--episodes", type=int, help="rollouts averaged per candidate")
    p.add_argument("--sigma", type=float)
    p.add_argument("--weight-decay", dest="weight_decay", type=float)
    p.add_argument("--generations", type=int)
    p.add_argument("--eval-every", dest="eval_every", type=int)
    p.add_argument("--eval-episodes", dest="eval_episodes", type=int)
    p.add_argument("--early-stop", dest="early_stop", type=int)
    p.add_argument("--mode", default="zh", choices=["zh", "z"], help="controller input: [z,h] (full) or z only")
    p.add_argument("--checkpoints", default="checkpoints")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = resolve(args, "controller")
    print("Controller config:", cfg, "| mode:", args.mode)

    run_dir = Path(f"runs/controller_{args.mode}")
    run_dir.mkdir(parents=True, exist_ok=True)
    es_path = run_dir / "es.pkl"
    rng = np.random.default_rng(args.seed)
    n_params = Controller(args.mode).num_params

    if args.resume and es_path.exists():
        with open(es_path, "rb") as f:
            es, start_gen, best_eval, rng = pickle.load(f)
        print(f"Resumed from generation {start_gen}")
    else:
        es = cma.CMAEvolutionStrategy(np.zeros(n_params), cfg["sigma"],
                                      {"popsize": cfg["popsize"], "seed": args.seed + 1, "verbose": -9})
        start_gen, best_eval = 0, -np.inf
        (run_dir / "log.csv").write_text("generation,mean,max,min,sigma,seconds\n")
        (run_dir / "eval.csv").write_text("generation,eval_mean,eval_std\n")
    print(f"{n_params} controller parameters, {args.workers} workers")

    with Pool(args.workers, initializer=init_worker,
              initargs=(args.checkpoints, args.mode, cfg["early_stop"])) as pool:
        for gen in range(start_gen + 1, cfg["generations"] + 1):
            t0 = time.time()
            solutions = es.ask()
            # Common random tracks for the whole population -> fairer comparison.
            seeds = rng.integers(0, EVAL_SEED_OFFSET, size=cfg["episodes"])
            jobs = [(s, seed, True) for s in solutions for seed in seeds]
            rewards = np.array(pool.map(play, jobs, chunksize=1)).reshape(len(solutions), cfg["episodes"])
            fitness = rewards.mean(axis=1)
            penalty = cfg["weight_decay"] * np.mean(np.square(solutions), axis=1)
            es.tell(solutions, (-(fitness - penalty)).tolist())  # CMA-ES minimises

            dt = time.time() - t0
            print(f"gen {gen:4d} | mean {fitness.mean():7.1f} | max {fitness.max():7.1f} | "
                  f"min {fitness.min():7.1f} | sigma {es.sigma:.4f} | {dt:4.0f}s", flush=True)
            with open(run_dir / "log.csv", "a") as f:
                f.write(f"{gen},{fitness.mean():.2f},{fitness.max():.2f},{fitness.min():.2f},{es.sigma:.5f},{dt:.1f}\n")

            if gen % cfg["eval_every"] == 0 or gen == cfg["generations"]:
                mean_params = es.mean.copy()
                eval_seeds = EVAL_SEED_OFFSET + np.arange(cfg["eval_episodes"])
                # Evaluation always plays full episodes, exactly like the paper.
                scores = np.array(pool.map(play, [(mean_params, s, False) for s in eval_seeds], chunksize=1))
                print(f"  >> eval gen {gen}: {scores.mean():.1f} +/- {scores.std():.1f} "
                      f"over {len(scores)} tracks", flush=True)
                with open(run_dir / "eval.csv", "a") as f:
                    f.write(f"{gen},{scores.mean():.2f},{scores.std():.2f}\n")
                if scores.mean() > best_eval:
                    best_eval = scores.mean()
                    np.save(run_dir / "best.npy", mean_params)
                    print(f"  >> new best controller saved ({best_eval:.1f})", flush=True)
            np.save(run_dir / "latest.npy", es.mean)
            with open(es_path, "wb") as f:
                pickle.dump((es, gen, best_eval, rng), f)


if __name__ == "__main__":
    main()
