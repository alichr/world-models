"""Step 4 - Train the M model (MDN-RNN) on latent sequences.

Learns P(z_{t+1} | z_t, a_t, h_t). The VAE is frozen; only data/series.npz is
used, so this trains in minutes.

    python scripts/04_train_mdrnn.py
    python scripts/04_train_mdrnn.py --steps 500        # quick test
Outputs: checkpoints/mdrnn.pt  and  runs/mdrnn/log.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from worldmodels.agent import get_device
from worldmodels.config import N_MIXTURES, PRESETS, RNN_HIDDEN, resolve
from worldmodels.data import SeriesDataset
from worldmodels.mdrnn import MDNRNN, mdn_loss


def to_device(batch, device):
    return [torch.from_numpy(np.ascontiguousarray(x)).float().to(device) for x in batch]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="laptop", choices=PRESETS)
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--seq-len", dest="seq_len", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--grad-clip", dest="grad_clip", type=float)
    p.add_argument("--series", default="data/series.npz")
    p.add_argument("--out", default="checkpoints/mdrnn.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = resolve(args, "mdrnn")
    print("MDN-RNN config:", cfg)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    n_eps = len(np.load(args.series)["lengths"])
    n_val = max(1, n_eps // 20)
    train = SeriesDataset(args.series, range(n_eps - n_val))
    val = SeriesDataset(args.series, range(n_eps - n_val, n_eps))
    print(f"{n_eps - n_val} train / {n_val} val episodes on {device}")

    rnn = MDNRNN(hidden=RNN_HIDDEN, n_mix=N_MIXTURES).to(device)
    opt = torch.optim.Adam(rnn.parameters(), lr=cfg["lr"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("runs/mdrnn").mkdir(parents=True, exist_ok=True)
    log = open("runs/mdrnn/log.csv", "w")
    log.write("step,train_nll,val_nll\n")

    t0, running = time.time(), []
    for step in range(1, cfg["steps"] + 1):
        z, a, target = to_device(train.sample_batch(cfg["batch_size"], cfg["seq_len"], rng), device)
        (logmix, mean, logstd), _ = rnn(z, a)
        loss = mdn_loss(logmix, mean, logstd, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), cfg["grad_clip"])
        opt.step()
        running.append(loss.item())

        if step % 100 == 0 or step == cfg["steps"]:
            with torch.no_grad():
                vz, va, vt = to_device(val.sample_batch(cfg["batch_size"], cfg["seq_len"], rng), device)
                val_nll = mdn_loss(*rnn(vz, va)[0], vt).item()
            train_nll = float(np.mean(running))
            running = []
            print(f"step {step:5d} | train NLL {train_nll:7.4f} | val NLL {val_nll:7.4f} | {time.time() - t0:5.0f}s",
                  flush=True)
            log.write(f"{step},{train_nll:.5f},{val_nll:.5f}\n")
            log.flush()
            torch.save({"model": rnn.state_dict(), "hidden": RNN_HIDDEN, "n_mix": N_MIXTURES, "step": step},
                       args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
