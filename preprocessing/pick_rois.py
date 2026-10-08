# -*- coding: utf-8 -*-
"""
pick_rois.py -- interactive 4x4 ROI picker on the rectified slice image (SPACE preprocessing).

Manuscript Methods section: "ROI definition"
Manuscript figure(s): Figure 1B (ROI boxes on the registered slice); input to every ROI-based module.

Standalone replacement for the picker that used to live inside core/roi_lfp_csd_metrics.py.
The slice image rectified to the electrode field (``data/masks/<slice>__rectified_grid.png``
written by preprocessing/tissue_registration_mask.py) is displayed over the 64x64 grid with the
tissue-contact mask outline; the user clicks the centre electrode of each ROI in the fixed order
M1, M2, S1, CA1, CA3, DG (the figure title says which one to click; nothing is typed). A 4x4 box
is built around each click (clamped inside the grid, same rule as roi_from_center() in
core/roi_lfp_csd_metrics.py).

Inputs
    data/masks/<slice>__contact_mask_64x64.npy      bool [64,64] tissue-contact mask
    data/masks/<slice>__rectified_grid.png          (default --image; PNG/JPG/TIFF accepted)
Outputs
    data/roi_definitions/<slice>_roi_definitions.csv   roi_index,roi_name,r0,r1,c0,c1  (r1/c1 exclusive)
    data/roi_definitions/<slice>_roi_overview.png      rectified image + mask outline + boxes

Usage
    python preprocessing/pick_rois.py --slice_id c57bl6_cor_6a            # interactive
    python preprocessing/pick_rois.py --slice_id c57bl6_cor_6a --render   # redraw overview from csv

Implementation notes
    * The rectified image spans exactly the electrode field, so it is drawn with
      extent=(0, W, H, 0) -- no cropping to the mask bounding box (the old embedded picker cropped
      to the bbox and then stretched it to the full grid, which mis-scaled the photo whenever the
      mask did not cover the whole array).
    * This is the one SPACE script that is interactive by design: the matplotlib backend is left
      at the platform default (TkAgg/QtAgg/macosx) so a window opens on Windows/macOS. ``Agg`` is
      used only with --render, which never needs a window.
    * If the image is missing, a blank grey background is used with a warning (mask only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- EDIT THIS ---
MASK_DIR = REPO_ROOT / "data" / "masks"
ROI_DIR = REPO_ROOT / "data" / "roi_definitions"
ROI_NAMES = "M1,M2,S1,CA1,CA3,DG"
ROI_SIZE = 4
GRID_H, GRID_W = 64, 64
# -----------------


def roi_from_center(r: int, c: int, H: int, W: int, size: int = 4) -> tuple[int, int, int, int]:
    """Fixed-size box centred on (r, c), shifted to stay inside the HxW grid. Returns r0, r1, c0, c1
    (Python slice bounds, r1/c1 exclusive). Identical to core/roi_lfp_csd_metrics.roi_from_center."""
    half_low = size // 2
    half_high = size - half_low
    r0, r1, c0, c1 = r - half_low, r + half_high, c - half_low, c + half_high
    if r0 < 0:
        r1 += -r0; r0 = 0
    if c0 < 0:
        c1 += -c0; c0 = 0
    if r1 > H:
        r0 -= r1 - H; r1 = H
    if c1 > W:
        c0 -= c1 - W; c1 = W
    return max(0, r0), min(H, r1), max(0, c0), min(W, c1)


def load_image(path: Path | None) -> np.ndarray | None:
    """Load PNG/JPG/TIFF as float [0,1] grayscale (or RGB for PNG/JPG). None if unavailable."""
    if path is None or not path.exists():
        print(f"[WARN] slice image not found ({path}); showing a blank grey background with the mask only.")
        return None
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            import tifffile
            img = tifffile.imread(str(path))
            if img.ndim == 3 and img.shape[-1] not in (3, 4):      # stack -> middle page
                img = img[img.shape[0] // 2]
            elif img.ndim == 4:
                img = img[0, img.shape[1] // 2]
            if img.ndim == 3:
                img = img[..., :3].mean(axis=-1)
        else:
            from PIL import Image
            img = np.asarray(Image.open(path).convert("RGB"))
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] could not read {path}: {e}; showing blank background.")
        return None
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img


def load_mask(mask_dir: Path, slice_id: str) -> np.ndarray:
    p = mask_dir / f"{slice_id}__contact_mask_64x64.npy"
    if not p.exists():
        raise FileNotFoundError(f"mask not found: {p}  (run preprocessing/tissue_registration_mask.py first)")
    m = np.load(p).astype(bool)
    if m.shape != (GRID_H, GRID_W):
        raise ValueError(f"mask {p} has shape {m.shape}, expected ({GRID_H},{GRID_W})")
    return m


def draw_background(ax, img: np.ndarray | None, mask: np.ndarray) -> None:
    """Rectified image aligned to the grid by extent, mask outline, faint electrode grid."""
    H, W = mask.shape
    if img is None:
        ax.imshow(np.full((H, W), 0.5), cmap="gray", vmin=0, vmax=1, extent=(0, W, H, 0))
    else:
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, extent=(0, W, H, 0), interpolation="bilinear")
    ax.imshow(np.ma.masked_where(~mask, mask.astype(float)), cmap="Blues", alpha=0.18, vmin=0, vmax=1,
              extent=(0, W, H, 0), interpolation="nearest")
    ax.contour(np.arange(W) + 0.5, np.arange(H) + 0.5, mask.astype(float), levels=[0.5], colors="black", linewidths=1.2)
    for i in range(0, H + 1, 8):
        ax.axhline(i, color="white", alpha=0.25, lw=0.5)
        ax.axvline(i, color="white", alpha=0.25, lw=0.5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xlabel("column (electrode)"); ax.set_ylabel("row (electrode)")


def draw_boxes(ax, rois: pd.DataFrame) -> None:
    import matplotlib.patches as patches
    colors = ["#ff7f0e", "#d62728", "#e377c2", "#1f77b4", "#2ca02c", "#9467bd"]
    for _, row in rois.iterrows():
        col = colors[int(row.roi_index) % len(colors)]
        ax.add_patch(patches.Rectangle((row.c0, row.r0), row.c1 - row.c0, row.r1 - row.r0,
                                       fill=False, edgecolor=col, lw=2))
        ax.text(row.c0, row.r0 - 0.6, str(row.roi_name), color=col, fontsize=10, fontweight="bold",
                ha="left", va="bottom", bbox=dict(facecolor="black", alpha=0.4, pad=1, lw=0))


def save_overview(out_png: Path, img: np.ndarray | None, mask: np.ndarray, rois: pd.DataFrame, slice_id: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_background(ax, img, mask)
    draw_boxes(ax, rois)
    ax.set_title(f"{slice_id}: {len(rois)} ROIs ({ROI_SIZE}x{ROI_SIZE})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] overview -> {out_png}")


def pick_interactively(img: np.ndarray | None, mask: np.ndarray, names: list[str], size: int) -> pd.DataFrame:
    """Open a window; one ginput click per ROI, prompt in the figure title. Boxes are drawn as you go."""
    import matplotlib.pyplot as plt
    H, W = mask.shape
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_background(ax, img, mask)
    rows = []
    for i, name in enumerate(names):
        ax.set_title(f"[{i + 1}/{len(names)}]  click the centre electrode of  {name}\n(close the window to abort)",
                     fontsize=11)
        fig.canvas.draw_idle()
        pts = plt.ginput(1, timeout=-1)
        if not pts:
            plt.close(fig)
            raise RuntimeError("ROI picking aborted (window closed before all ROIs were clicked). Nothing written.")
        x, y = pts[0]
        r, c = int(np.clip(np.floor(y), 0, H - 1)), int(np.clip(np.floor(x), 0, W - 1))
        r0, r1, c0, c1 = roi_from_center(r, c, H, W, size)
        rows.append(dict(roi_index=i, roi_name=name, r0=r0, r1=r1, c0=c0, c1=c1))
        print(f"  [ROI] {name}: centre (row={r}, col={c}) -> rows {r0}:{r1}, cols {c0}:{c1}")
        draw_boxes(ax, pd.DataFrame(rows[-1:]))   # show the box just placed
    ax.set_title("done - closing")
    fig.canvas.draw_idle()
    plt.pause(0.5)
    plt.close(fig)
    plt.ioff()
    return pd.DataFrame(rows, columns=["roi_index", "roi_name", "r0", "r1", "c0", "c1"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice_id", required=True, help="slice id = prefix of <slice>__contact_mask_64x64.npy")
    p.add_argument("--mask_dir", default=str(MASK_DIR))
    p.add_argument("--roi_dir", default=str(ROI_DIR))
    p.add_argument("--image", default=None, help="slice image rectified to the electrode field "
                                                "(default: <mask_dir>/<slice>__rectified_grid.png; PNG/JPG/TIFF)")
    p.add_argument("--names", default=ROI_NAMES, help="comma-separated ROI names in click order")
    p.add_argument("--size", type=int, default=ROI_SIZE, help="ROI box side (electrodes)")
    p.add_argument("--redo", action="store_true", help="overwrite an existing roi_definitions.csv")
    p.add_argument("--render", "--image_only", dest="render", action="store_true",
                   help="no clicking: redraw the overview PNG from the existing csv (uses the Agg backend)")
    args = p.parse_args(argv)

    if args.render:
        import matplotlib
        matplotlib.use("Agg")

    mask_dir, roi_dir = Path(args.mask_dir), Path(args.roi_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)
    out_csv = roi_dir / f"{args.slice_id}_roi_definitions.csv"
    out_png = roi_dir / f"{args.slice_id}_roi_overview.png"
    img_path = Path(args.image) if args.image else mask_dir / f"{args.slice_id}__rectified_grid.png"

    mask = load_mask(mask_dir, args.slice_id)
    img = load_image(img_path)

    if args.render:
        if not out_csv.exists():
            print(f"[ERROR] {out_csv} does not exist; run without --render to pick ROIs first.")
            return 2
        rois = pd.read_csv(out_csv)
        save_overview(out_png, img, mask, rois, args.slice_id)
        return 0

    if out_csv.exists() and not args.redo:
        print(f"[SKIP] {out_csv} already exists (use --redo to overwrite, --render to redraw the overview).")
        return 0

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    print(f"[pick_rois] {args.slice_id}: click {len(names)} ROI centres in this order: {', '.join(names)}")
    rois = pick_interactively(img, mask, names, args.size)
    rois.to_csv(out_csv, index=False)
    print(f"[OK] ROI definitions -> {out_csv}")
    save_overview(out_png, img, mask, rois, args.slice_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
