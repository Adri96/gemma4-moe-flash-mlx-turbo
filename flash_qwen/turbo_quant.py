"""TurboQuant KV cache compression (PolarQuant-only, no QJL).

Implements the rotation + per-coordinate scalar quantization scheme from
"TurboQuant: Online Vector Quantization" (Google, ICLR 2026, arXiv 2504.19874).
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_lm.models.base import create_causal_mask


# ---------------------------------------------------------------------------
# Lloyd-Max codebook for N(0, 1) — computed once at init via pure numpy
# ---------------------------------------------------------------------------

def _gaussian_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _gaussian_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lloyd_max_gaussian(n_levels: int, max_iter: int = 200, tol: float = 1e-10):
    """Compute optimal Lloyd-Max centroids and decision boundaries for N(0,1).

    Returns:
        centroids: (n_levels,) float32 — reconstruction values
        boundaries: (n_levels - 1,) float32 — decision thresholds for quantization
    """
    # Initialize centroids uniformly over [-3.5, 3.5]
    edges = np.linspace(-3.5, 3.5, n_levels + 1)
    centroids = ((edges[:-1] + edges[1:]) / 2).astype(np.float64)

    for _ in range(max_iter):
        # Decision boundaries = midpoints between adjacent centroids
        bounds = np.empty(n_levels + 1, dtype=np.float64)
        bounds[0] = -np.inf
        bounds[-1] = np.inf
        bounds[1:-1] = (centroids[:-1] + centroids[1:]) / 2

        # Update centroids = E[X | X in partition_i] for X ~ N(0,1)
        # E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
        new = np.empty_like(centroids)
        for i in range(n_levels):
            lo, hi = float(bounds[i]), float(bounds[i + 1])
            prob = _gaussian_cdf(hi) - _gaussian_cdf(lo)
            if prob > 1e-15:
                new[i] = (_gaussian_pdf(lo) - _gaussian_pdf(hi)) / prob
            else:
                new[i] = 0.0 if not (math.isfinite(lo) and math.isfinite(hi)) else (lo + hi) / 2
        if np.max(np.abs(new - centroids)) < tol:
            centroids = new
            break
        centroids = new

    decision_boundaries = ((centroids[:-1] + centroids[1:]) / 2).astype(np.float32)
    return centroids.astype(np.float32), decision_boundaries


# ---------------------------------------------------------------------------
# Random orthogonal rotation via QR
# ---------------------------------------------------------------------------

def generate_rotation(dim: int, seed: int = 42) -> mx.array:
    """Haar-distributed random orthogonal matrix via QR decomposition."""
    key = mx.random.key(seed)
    G = mx.random.normal(shape=(dim, dim), key=key)
    mx.eval(G)
    cpu = mx.cpu
    Q, R = mx.linalg.qr(G, stream=cpu)
    mx.eval(Q, R)
    d = mx.sign(mx.diag(R))
    return Q * d[None, :]


# ---------------------------------------------------------------------------
# TurboQuantCache — drop-in replacement for mlx_lm KVCache
# ---------------------------------------------------------------------------

class TurboQuantCache:
    """KV cache that stores quantized indices + per-token scales.

    Quantize-on-store, dequantize-on-fetch.  No ``bits`` attribute so
    ``scaled_dot_product_attention`` takes the standard (non-quantized) path.
    """

    step = 256  # allocation chunk size (tokens)

    def __init__(self, head_dim: int = 256, bits: int = 4, seed: int = 42):
        self.head_dim = head_dim
        self._bits = bits
        self.n_centroids = 1 << bits

        # Precompute rotation matrix  (D, D)
        self.rotation = generate_rotation(head_dim, seed)

        # Precompute Lloyd-Max codebook for N(0, 1)
        c_np, b_np = lloyd_max_gaussian(self.n_centroids)
        self.centroids = mx.array(c_np)   # (C,)
        self.boundaries = mx.array(b_np)  # (C-1,)

        # Compressed storage — lazily allocated on first update
        self._key_idx: Optional[mx.array] = None   # uint8  (B, H, alloc, D)
        self._val_idx: Optional[mx.array] = None
        self._key_scl: Optional[mx.array] = None   # float16 (B, H, alloc, 1)
        self._val_scl: Optional[mx.array] = None
        # Decompressed buffer — avoids full re-dequant every step
        self._deq_keys: Optional[mx.array] = None  # (B, H, T, D)
        self._deq_vals: Optional[mx.array] = None
        self.offset: int = 0
        self._dtype = mx.bfloat16  # output dtype, captured from first input

    # ---- quantize / dequantize -------------------------------------------

    def _quantize(self, x: mx.array):
        """x: (B, H, L, D) → indices (uint8), scale (float16)."""
        rotated = x @ self.rotation
        scale = mx.sqrt((rotated * rotated).mean(axis=-1, keepdims=True))
        scale = mx.maximum(scale, 1e-8)
        normed = rotated / scale
        idx = (normed[..., None] > self.boundaries).sum(axis=-1).astype(mx.uint8)
        return idx, scale.astype(mx.float16)

    def _dequantize(self, idx: mx.array, scl: mx.array) -> mx.array:
        """indices + scales → reconstructed tensor in original dtype."""
        recon = self.centroids[idx] * scl
        return (recon @ self.rotation.T).astype(self._dtype)

    # ---- KVCache-compatible API ------------------------------------------

    def _grow(self, B: int, H: int, D: int, need: int):
        """Pre-allocate compressed + decompressed buffers in chunks."""
        alloc = ((need + self.step - 1) // self.step) * self.step
        prev = self.offset

        new_ki = mx.zeros((B, H, alloc, D), dtype=mx.uint8)
        new_vi = mx.zeros((B, H, alloc, D), dtype=mx.uint8)
        new_ks = mx.zeros((B, H, alloc, 1), dtype=mx.float16)
        new_vs = mx.zeros((B, H, alloc, 1), dtype=mx.float16)
        new_dk = mx.zeros((B, H, alloc, D), dtype=self._dtype)
        new_dv = mx.zeros((B, H, alloc, D), dtype=self._dtype)

        if self._key_idx is not None and prev > 0:
            new_ki = mx.concatenate([self._key_idx[..., :prev, :], new_ki[..., prev:, :]], axis=2)
            new_vi = mx.concatenate([self._val_idx[..., :prev, :], new_vi[..., prev:, :]], axis=2)
            new_ks = mx.concatenate([self._key_scl[..., :prev, :], new_ks[..., prev:, :]], axis=2)
            new_vs = mx.concatenate([self._val_scl[..., :prev, :], new_vs[..., prev:, :]], axis=2)
            new_dk = mx.concatenate([self._deq_keys[..., :prev, :], new_dk[..., prev:, :]], axis=2)
            new_dv = mx.concatenate([self._deq_vals[..., :prev, :], new_dv[..., prev:, :]], axis=2)

        self._key_idx, self._val_idx = new_ki, new_vi
        self._key_scl, self._val_scl = new_ks, new_vs
        self._deq_keys, self._deq_vals = new_dk, new_dv

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new K/V (quantized) and return full dequantized cache."""
        B, H, L, D = keys.shape
        self._dtype = keys.dtype
        prev = self.offset
        need = prev + L

        k_idx, k_scl = self._quantize(keys)
        v_idx, v_scl = self._quantize(values)

        if self._key_idx is None or need > self._key_idx.shape[2]:
            self._grow(B, H, D, need)

        # Write compressed
        self._key_idx[..., prev:need, :] = k_idx
        self._val_idx[..., prev:need, :] = v_idx
        self._key_scl[..., prev:need, :] = k_scl
        self._val_scl[..., prev:need, :] = v_scl

        # Dequantize only new tokens, write into pre-allocated buffer
        self._deq_keys[..., prev:need, :] = self._dequantize(k_idx, k_scl)
        self._deq_vals[..., prev:need, :] = self._dequantize(v_idx, v_scl)
        self.offset = need

        return self._deq_keys[..., :self.offset, :], self._deq_vals[..., :self.offset, :]

    def make_mask(self, N: int, return_array: bool = False,
                  window_size: Optional[int] = None):
        if N == 1:
            return None
        if return_array or (window_size and N > window_size):
            return create_causal_mask(N, window_size=window_size)
        return "causal"

    def empty(self) -> bool:
        return self._key_idx is None

    @property
    def state(self):
        if self._deq_keys is None:
            return None, None
        return self._deq_keys, self._deq_vals

    @state.setter
    def state(self, v):
        keys, values = v
        if keys is None:
            self._key_idx = self._val_idx = None
            self._key_scl = self._val_scl = None
            self._deq_keys = self._deq_vals = None
            self.offset = 0
            return
        self._dtype = keys.dtype
        self._key_idx, self._key_scl = self._quantize(keys)
        self._val_idx, self._val_scl = self._quantize(values)
        self._deq_keys = keys
        self._deq_vals = values
        self.offset = keys.shape[2]

    @property
    def nbytes(self) -> int:
        """Compressed bytes in use (indices + scales)."""
        if self._key_idx is None:
            return 0
        n = self.offset
        D = self.head_dim
        B = self._key_idx.shape[0]
        H = self._key_idx.shape[1]
        # indices: uint8 → 1 byte/coord, scales: float16 → 2 bytes/token
        return B * H * n * (D * 1 + 1 * 2) * 2  # ×2 for K and V

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        if self._deq_keys is not None:
            self._deq_keys = self._deq_keys[..., n:, :]
            self._deq_vals = self._deq_vals[..., n:, :]
        return n
