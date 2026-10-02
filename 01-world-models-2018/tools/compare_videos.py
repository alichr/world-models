"""Stack two evaluation videos side by side with labels (e.g. untrained vs trained).

    python tools/compare_videos.py runs/snapshots/untrained_drive.mp4 "UNTRAINED controller" \
                             runs/snapshots/gen30_drive.mp4 "TRAINED controller (gen 30)" \
                             --out runs/snapshots/untrained_vs_gen30.mp4
"""

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

GAME_WIDTH = 600  # left part of 06_evaluate.py videos; the VAE panels are dropped


def label_bar(text, color):
    im = Image.new("RGB", (GAME_WIDTH, 32), color)
    ImageDraw.Draw(im).text((10, 9), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("left_video")
    p.add_argument("left_label")
    p.add_argument("right_video")
    p.add_argument("right_label")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    left, right = imageio.mimread(args.left_video, memtest=False), imageio.mimread(args.right_video, memtest=False)
    bars = label_bar(args.left_label, (120, 30, 30)), label_bar(args.right_label, (30, 90, 40))
    writer = imageio.get_writer(args.out, fps=50, quality=8, macro_block_size=8)
    for i in range(max(len(left), len(right))):
        a = left[min(i, len(left) - 1)][:400, :GAME_WIDTH]    # freeze on the last frame when an episode ends early
        b = right[min(i, len(right) - 1)][:400, :GAME_WIDTH]
        gap = np.full((432, 8, 3), 255, np.uint8)
        writer.append_data(np.concatenate([np.concatenate([bars[0], a]), gap, np.concatenate([bars[1], b])], 1))
    writer.close()
    print("saved", args.out)


if __name__ == "__main__":
    main()
