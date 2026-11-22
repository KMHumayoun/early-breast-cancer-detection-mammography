from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_image(path, height):
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    photo = ds.PhotometricInterpretation
    if pixels.ndim != 2 or photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("Expected a single grayscale mammogram")
    valid = np.ones(pixels.shape, dtype=bool)
    if "PixelPaddingValue" in ds:
        start = int(ds.PixelPaddingValue)
        end = int(ds.get("PixelPaddingRangeLimit", start))
        valid &= (pixels < min(start, end)) | (pixels > max(start, end))
    if not valid.any():
        raise ValueError("Image contains only padding")
    # Estimate display limits on a regular sample to avoid sorting a 12 MP image.
    sample = pixels[::4, ::4][valid[::4, ::4]]
    if sample.size == 0:
        sample = pixels[valid]
    lo, hi = np.percentile(sample, [0.5, 99.8])
    if hi <= lo:
        lo, hi = float(pixels[valid].min()), float(pixels[valid].max())
    if hi <= lo:
        raise ValueError("Uniform mammogram")
    grayscale = np.clip((pixels.astype(np.float32) - lo) / (hi - lo), 0, 1)
    if photo == "MONOCHROME1":
        grayscale = 1 - grayscale
    grayscale[~valid] = 0
    image = Image.fromarray(np.rint(grayscale * 255).astype(np.uint8))
    width = round(height * 2 / 3)
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height))
    canvas.paste(image, ((width-image.width)//2, (height-image.height)//2))
    return torch.from_numpy(np.asarray(canvas).copy()).float().unsqueeze(0).div_(255)


class BreastImages(Dataset):
    def __init__(self, breasts, root, height, views):
        self.rows = breasts
        self.root = Path(root)
        self.height = height
        self.views = views

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        pixels = torch.stack([read_image(self.root / row[v], self.height) for v in self.views])
        return pixels, torch.tensor(row["cancer"], dtype=torch.float32)
