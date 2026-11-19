"""Contrast Limited Adaptive Histogram Equalization (CLAHE) untuk CXR.

Dipakai hanya pada eksperimen ablation (config preprocessing.clahe: true).
CLAHE diterapkan pada kanal keabuan setelah pad-to-square dan resize, lalu
dikembalikan ke 3 kanal agar kompatibel dengan backbone pralatih ImageNet.
Parameter dipilih a priori (clipLimit 2,0; grid 8x8; nilai umum di literatur
CXR) dan tidak disetel berdasarkan hasil validation maupun test.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8, 8)


class CLAHE:
    def __init__(self, clip_limit: float = CLIP_LIMIT, tile_grid_size=TILE_GRID_SIZE):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)

    def __call__(self, image: Image.Image) -> Image.Image:
        # Objek cv2 dibuat per panggilan agar aman di DataLoader multi-worker.
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        return Image.fromarray(clahe.apply(gray)).convert("RGB")

    def __repr__(self) -> str:
        return f"CLAHE(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})"
