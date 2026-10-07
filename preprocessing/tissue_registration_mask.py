# -*- coding: utf-8 -*-
"""
tissue_registration_mask.py -- SPACE framework: optical registration and tissue-contact mask.

Manuscript figure(s): Figure 1A (optical registration, grid mapping, tissue segmentation),
Supplementary Figure 1 (tissue-contact mask).

What it computes:
    Interactive optical registration of a slice bright-field photo onto the HD-MEA's
    64x64 electrode grid. The user clicks the 4 corners of the MEA field (TL, TR, BR,
    BL); the image is homography-warped onto a square canvas aligned with the grid,
    Otsu-thresholded (with adaptive/percentile fallbacks) to segment tissue from
    background, and the resulting tissue mask is binned down to a 64x64 boolean contact
    mask by per-cell coverage fraction. This mask is consumed by every downstream
    analysis that maps a channel vector back onto the 64x64 electrode grid.

Inputs / outputs (see data/README.md for the full schema):
    Input:  a bright-field photo of the slice under ``data/slice_photos/``.
    Output: written under ``data/masks/``, named ``<slice_id>__contact_mask_64x64.npy``
            (the boolean mask consumed by ``preprocessing/brw_to_2ms_h5.py`` and other
            downstream scripts), plus ``<slice_id>__rectified_grid.png``,
            ``<slice_id>__contact_overlay.png``, ``<slice_id>__contact_mask_64x64.png``,
            and a ``<slice_id>__contact_manifest.json`` recording the click coordinates,
            homography, and thresholds used.

Deps: numpy, opencv-python, matplotlib, pillow, scipy

Usage:
  python tissue_registration_mask.py --image /path/to/slice.bmp --out out_dir
  # optional knobs:
  # --canvas 1024 --coverage 0.30 --percentile 60
"""
import os, json, argparse
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, uniform_filter

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _robust_gray(arr):
    # robust contrast stretch
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1: p99 = p1 + 1e-3
    arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
    return arr

def _detect_contact_mask(gray01, method="otsu"):
    import cv2
    img8 = (gray01 * 255).astype(np.uint8)

    if method == "otsu":
        _, m = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif method == "adaptive":
        m = cv2.adaptiveThreshold(img8, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,
                                  51, 2)
    else:  # fallback percentile
        th = np.percentile(gray01, 75)
        m = (gray01 > th).astype(np.uint8) * 255

    # Morphology cleanup
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

    # Keep largest connected region only
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (labels == largest).astype(np.uint8) * 255

    return m > 0

def _bin_to_64x64(mask_rectified, coverage=0.6):
    H, W = mask_rectified.shape
    ys = np.linspace(0, H, 65).astype(int)
    xs = np.linspace(0, W, 65).astype(int)
    out = np.zeros((64,64), dtype=np.uint8)
    for j in range(64):
        for i in range(64):
            tile = mask_rectified[ys[j]:ys[j+1], xs[i]:xs[i+1]]
            if tile.size > 0 and tile.mean() > coverage:
                out[j,i] = 1
    return out


def _grid_overlay(img_bgr, n=64, color=(0,255,0), thickness=1):
    h, w = img_bgr.shape[:2]
    step_x = w / n; step_y = h / n
    out = img_bgr.copy()
    for i in range(1, n):
        x = int(round(i * step_x))
        y = int(round(i * step_y))
        cv2.line(out, (x,0), (x,h-1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(out, (0,y), (w-1,y), color, thickness, lineType=cv2.LINE_AA)
    return out

def _bin_to_64x64(mask_rectified, coverage=0.30):
    H, W = mask_rectified.shape
    assert H == W, "Rectified canvas must be square"
    cell = H // 64  # integer cell size
    # If not divisible, use linspace edges for precise binning
    ys = np.linspace(0, H, 65).astype(int)
    xs = np.linspace(0, W, 65).astype(int)
    out = np.zeros((64,64), dtype=np.uint8)
    for j in range(64):
        for i in range(64):
            y0, y1 = ys[j], ys[j+1]
            x0, x1 = xs[i], xs[i+1]
            tile = mask_rectified[y0:y1, x0:x1]
            if tile.size == 0: 
                out[j,i] = 0
            else:
                frac = tile.mean()
                out[j,i] = 1 if frac >= coverage else 0
    return out

def pick_four_corners(img_rgb):
    """
    Matplotlib-based clicking: TL -> TR -> BR -> BL
    Keys:
      - 'r' to reset clicks
      - 'enter'/'return' to accept when 4 points chosen
      - close the window to cancel
    """
    pts = []

    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(img_rgb)
    ax.set_title("Click corners in order: TL → TR → BR → BL\n(r = reset, Enter = accept)")
    scat = ax.scatter([], [], c="r", s=60)

    ax_reset = plt.axes([0.8, 0.01, 0.15, 0.06])
    breset = Button(ax_reset, 'Reset')

    def onclick(event):
        if event.inaxes != ax: 
            return
        if len(pts) < 4:
            pts.append([event.xdata, event.ydata])
            scat.set_offsets(np.array(pts))
            fig.canvas.draw_idle()

    def onkey(event):
        if event.key == 'enter' and len(pts) == 4:
            plt.close(fig)
        if event.key == 'r':
            pts.clear()
            scat.set_offsets(np.zeros((0,2)))
            fig.canvas.draw_idle()

    def onreset(event):
        pts.clear()
        scat.set_offsets(np.zeros((0,2)))
        fig.canvas.draw_idle()

    cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
    cid_key   = fig.canvas.mpl_connect('key_press_event', onkey)
    breset.on_clicked(onreset)
    plt.show()

    if len(pts) != 4:
        raise RuntimeError("Cancelled or fewer than 4 points selected.")
    # enforce TL, TR, BR, BL order if user misclicked slightly:
    pts = np.array(pts, dtype=np.float32)
    # heuristic: sort by y, then split top/bottom; sort each by x
    idx = np.argsort(pts[:,1])
    top = pts[idx[:2]][np.argsort(pts[idx[:2],0])]
    bot = pts[idx[2:]][np.argsort(pts[idx[2:],0])]
    ordered = np.vstack([top, bot])  # TL, TR, BL?, BR? -> fix:
    # Ensure order TL, TR, BR, BL
    TL, TR, B1, B2 = ordered
    # Decide BR vs BL by x
    BR, BL = (B2, B1) if B2[0] > B1[0] else (B1, B2)
    return np.array([TL, TR, BR, BL], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser(description="Click corners → rectify → contact mask → 64x64 array")
    ap.add_argument("--image", required=True, help="Path to slice photo")
    ap.add_argument("--out", default="contact_mask_out", help="Output folder")
    ap.add_argument("--canvas", type=int, default=1024, help="Rectified canvas size (square)")
    ap.add_argument("--percentile", type=float, default=60.0, help="Threshold percentile for contact mask")
    ap.add_argument("--coverage", type=float, default=0.30, help="Min fraction per cell to mark as 1")
    args = ap.parse_args()

    outdir = _ensure_dir(args.out)
    base = os.path.splitext(os.path.basename(args.image))[0]

    # 1) Load image
    pil = Image.open(args.image).convert("RGB")
    rgb = np.array(pil)
    gray = np.array(ImageOps.grayscale(pil)).astype(np.float32)

    # 2) Click four corners (TL, TR, BR, BL)
    print("👉 Select 4 corners (TL, TR, BR, BL). Press 'r' to reset, Enter to accept.")
    quad = pick_four_corners(rgb)  # (4,2)

    # 3) Warp to square canvas aligned with the grid
    N = args.canvas
    dst = np.array([[0,0], [N-1,0], [N-1,N-1], [0,N-1]], dtype=np.float32)
    H, _ = cv2.findHomography(quad, dst, method=cv2.RANSAC)
    rect_rgb  = cv2.warpPerspective(rgb,  H, (N, N))
    rect_gray = cv2.warpPerspective(gray, H, (N, N))
    rect_gray01 = _robust_gray(rect_gray)

    # 4) Detect contact/tissue mask on rectified image
    contact = _detect_contact_mask(rect_gray01)

    # 5) Down-bin to 64×64 via per-cell coverage
    contact_64 = _bin_to_64x64(contact, coverage=args.coverage).astype(np.uint8)

    # 6) Save images & arrays
    # 6a) Rectified image with 64×64 grid overlay
    rect_bgr = cv2.cvtColor(rect_rgb, cv2.COLOR_RGB2BGR)
    grid_img = _grid_overlay(rect_bgr, n=64, color=(0,255,0), thickness=1)
    cv2.imwrite(os.path.join(outdir, f"{base}__rectified_grid.png"), grid_img)

    # 6b) Contact overlay
    overlay = rect_rgb.copy()
    overlay[contact.astype(bool)] = (overlay[contact.astype(bool)] * 0.5 + np.array([255,0,0])*0.5).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(outdir, f"{base}__contact_overlay.png"))

    # 6c) 64×64 mask visualisation
    vis64 = (contact_64 * 255).astype(np.uint8)
    vis64 = cv2.resize(vis64, (N, N), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(outdir, f"{base}__contact_mask_64x64.png"), vis64)

    # 6d) Save arrays
    np.save(os.path.join(outdir, f"{base}__contact_mask_64x64.npy"), contact_64)
    np.save(os.path.join(outdir, f"{base}__homography.npy"), H)

    # 6e) Manifest
    manifest = {
        "image": os.path.abspath(args.image),
        "outdir": os.path.abspath(outdir),
        "canvas_px": int(N),
        "clicked_corners_xy": quad.tolist(),   # TL, TR, BR, BL
        "homography_dst_square": [0,0,N-1,N-1],
        "percentile_threshold": float(args.percentile),
        "coverage_threshold": float(args.coverage),
        "contact_mask_64x64_npy": os.path.abspath(os.path.join(outdir, f"{base}__contact_mask_64x64.npy")),
        "rectified_grid_png": os.path.abspath(os.path.join(outdir, f"{base}__rectified_grid.png")),
        "contact_overlay_png": os.path.abspath(os.path.join(outdir, f"{base}__contact_overlay.png")),
        "contact_mask_64x64_png": os.path.abspath(os.path.join(outdir, f"{base}__contact_mask_64x64.png")),
    }
    with open(os.path.join(outdir, f"{base}__contact_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("✓ Saved:")
    print("  -", manifest["rectified_grid_png"])
    print("  -", manifest["contact_overlay_png"])
    print("  -", manifest["contact_mask_64x64_png"])
    print("  -", manifest["contact_mask_64x64_npy"])
    print("  -", os.path.join(outdir, f"{base}__contact_manifest.json"))
    print("\nNext: load the .npy as CONTACT_ROI and intersect with GOOD_MASK in your H5 before analysis.")
    # The downstream extractor/analysis scripts all expect a 64x64 grid + boolean mask.

if __name__ == "__main__":
    main()