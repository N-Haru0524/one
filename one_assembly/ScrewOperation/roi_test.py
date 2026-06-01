"""Interactive ROI inspector for ScrewOperation image pairs.

Opens an OpenCV window with a frame slider over the cam1/cam2 PNGs in a
dataset directory, overlays the ROI rectangles from ScrewConfig (or
overrides via --roi1 / --roi2), and shows the cropped windows in a second
viewer. Useful for sanity-checking that the ROI boxes line up with the
screw / bit before training.

Usage:
    uv run python -m one_assembly.ScrewOperation.roi_test \\
        --data_dir <ep_dir> [--roi1 X1 Y1 X2 Y2] [--roi2 X1 Y1 X2 Y2]

Controls inside the cv2 window: SPACE = play/pause, q = quit.
Requires an X11 display.
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image

from one_assembly.ScrewOperation.config import ScrewConfig, load_config, merge_cli_args
from one_assembly.ScrewOperation.preprocess import apply_roi, rotate_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--roi1", type=int, nargs=4, default=None)
    parser.add_argument("--roi2", type=int, nargs=4, default=None)
    parser.add_argument("--rotate1", type=int, choices=(0, 90, 180, 270), default=None,
                        help="CW rotation (deg) applied to cam1 before ROI; overrides config")
    parser.add_argument("--rotate2", type=int, choices=(0, 90, 180, 270), default=None,
                        help="CW rotation (deg) applied to cam2 before ROI; overrides config")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    config_path = os.path.join(args.data_dir, "config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = ScrewConfig()
    config = merge_cli_args(config, args)

    image_dir = os.path.join(args.data_dir, "images")
    cam1_files = sorted(glob.glob(os.path.join(image_dir, "*_cam1.png")))
    cam2_files = sorted(glob.glob(os.path.join(image_dir, "*_cam2.png")))

    if not cam1_files:
        print(f"No images found in {image_dir}")
        return

    n = min(len(cam1_files), len(cam2_files))
    print(f"Found {n} image pairs")
    print(f"rotate1: {config.rotate1}° CW   ROI1: {config.roi1}")
    print(f"rotate2: {config.rotate2}° CW   ROI2: {config.roi2}")
    print("Controls: SPACE=play/pause, q=quit")

    window_name = "ROI Test (SPACE: play/pause, q: quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", window_name, 0, n - 1, lambda x: None)

    playing = False
    prev_idx = -1
    wait_ms = int(1000 / args.fps)

    while True:
        idx = cv2.getTrackbarPos("frame", window_name)

        if playing:
            idx = (idx + 1) % n
            cv2.setTrackbarPos("frame", window_name, idx)

        if idx != prev_idx:
            prev_idx = idx

            # Open raw -> rotate (CW) -> ROI crop. The displayed "raw" view shows
            # the ROTATED frame so ROI coords match what the model sees.
            img1 = rotate_image(Image.open(cam1_files[idx]).convert("RGB"), int(config.rotate1))
            img2 = rotate_image(Image.open(cam2_files[idx]).convert("RGB"), int(config.rotate2))

            img1_raw = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_raw = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

            x1, y1, x2, y2 = config.roi1
            cv2.rectangle(img1_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x1, y1, x2, y2 = config.roi2
            cv2.rectangle(img2_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img1_crop = np.array(apply_roi(img1, config.roi1))
            img2_crop = np.array(apply_roi(img2, config.roi2))
            img1_crop = cv2.cvtColor(img1_crop, cv2.COLOR_RGB2BGR)
            img2_crop = cv2.cvtColor(img2_crop, cv2.COLOR_RGB2BGR)

            # Display cam2 on the LEFT and cam1 on the RIGHT to match the rig's
            # physical layout.
            raw = np.hstack((img2_raw, img1_raw))
            crop = np.hstack((img2_crop, img1_crop))

            status = "PLAY" if playing else "PAUSE"
            label = f"[{idx:04d}/{n - 1}] {status}"
            cv2.putText(raw, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, raw)
            cv2.imshow("Cropped ROI", crop)

        key = cv2.waitKey(wait_ms if playing else 30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
