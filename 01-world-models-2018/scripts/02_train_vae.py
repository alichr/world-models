"""Step 2 - Train the V model (VAE) on individual frames.

The VAE is trained on its own: it never sees actions or rewards, only frames.

    python scripts/02_train_vae.py                    # laptop preset
    python scripts/02_train_vae.py --epochs 1         # quick test
Outputs: checkpoints/vae.pt  and  runs/vae/epoch_XX.png (input vs reconstruction)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldmodels.agent import get_device
from worldmodels.config import PRESETS, Z_SIZE, resolve
from worldmodels.data import iterate_frame_batches, list_rollouts, split_files
from worldmodels.env import to_tensor_frames
from worldmodels.vae import VAE, vae_loss


@torch.no_grad()
def evaluate(vae, files, device, batch_size):
    vae.eval()
    totals = np.zeros(3)
    n = 0
    for frames in iterate_frame_batches(files, batch_size, np.random.default_rng(0)):
        x = to_tensor_frames(frames).to(device)
        recon, mu, logvar = vae(x)
        totals += [t.item() for t in vae_loss(recon, x, mu, logvar)]
        n += 1
    vae.train()
    return totals / max(n, 1)


@torch.no_grad()
def save_reconstructions(vae, files, device, path, n=8):
    frames = np.load(files[0])["obs"][::max(1, len(np.load(files[0])["obs"]) // n)][:n]
    x = to_tensor_frames(frames).to(device)
    mu, _ = vae.encode(x)
    recon = vae.decode(mu)
    grid = torch.cat([x, recon]).permute(0, 2, 3, 1).cpu().numpy()  # 2n images
    top, bottom = np.concatenate(grid[:n], 1), np.concatenate(grid[n:], 1)
    Image.fromarray((np.concatenate([top, bottom], 0) * 255).astype(np.uint8)).resize(
        (n * 128, 256), Image.NEAREST).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="laptop", choices=PRESETS)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--kl-tolerance", dest="kl_tolerance", type=float)
    p.add_argument("--data", default="data/rollouts")
    p.add_argument("--out", default="checkpoints/vae.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = resolve(args, "vae")
    print("VAE config:", cfg)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = get_device()
    train_files, val_files = split_files(list_rollouts(args.data))
    print(f"{len(train_files)} train / {len(val_files)} val rollouts on {device}")

    vae = VAE(Z_SIZE).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg["lr"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("runs/vae").mkdir(parents=True, exist_ok=True)
    log = open("runs/vae/log.csv", "w")
    log.write("epoch,step,train_loss,train_recon,train_kl,val_loss,val_recon,val_kl\n")

    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        t0, epoch_sum, epoch_n = time.time(), torch.zeros(3, device=device), 0
        for frames in iterate_frame_batches(train_files, cfg["batch_size"], rng):
            x = to_tensor_frames(frames).to(device)
            recon, mu, logvar = vae(x)
            loss, r_loss, kl = vae_loss(recon, x, mu, logvar, cfg["kl_tolerance"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            epoch_sum += torch.stack([loss.detach(), r_loss.detach(), kl.detach()])
            epoch_n += 1
            if step % 500 == 0:
                avg = (epoch_sum / epoch_n).tolist()
                print(f"epoch {epoch} step {step:6d} | loss {avg[0]:8.2f} recon {avg[1]:8.2f} kl {avg[2]:6.2f}"
                      f" | {time.time() - t0:5.0f}s", flush=True)
        train = (epoch_sum / max(epoch_n, 1)).tolist()
        val = evaluate(vae, val_files, device, cfg["batch_size"])
        print(f"== epoch {epoch} done in {time.time() - t0:.0f}s | val loss {val[0]:.2f} "
              f"(recon {val[1]:.2f}, kl {val[2]:.2f})", flush=True)
        log.write(f"{epoch},{step},{','.join(f'{v:.4f}' for v in train)},{','.join(f'{v:.4f}' for v in val)}\n")
        log.flush()
        torch.save({"model": vae.state_dict(), "z_size": Z_SIZE, "epoch": epoch}, args.out)
        save_reconstructions(vae, val_files, device, f"runs/vae/epoch_{epoch:02d}.png")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
