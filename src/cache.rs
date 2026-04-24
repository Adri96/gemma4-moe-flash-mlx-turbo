use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::perf::PerfStats;

/// Generate normalized Hadamard matrix H_n (Sylvester construction).
/// H * H^T = I. n must be a power of 2.
fn generate_hadamard(n: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; n * n];
    h[0] = 1.0;
    let mut size = 1;
    while size < n {
        for i in 0..size {
            for j in 0..size {
                let val = h[i * n + j];
                h[i * n + (j + size)] = val;
                h[(i + size) * n + j] = val;
                h[(i + size) * n + (j + size)] = -val;
            }
        }
        size *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for val in &mut h {
        *val *= scale;
    }
    h
}

/// Deterministic ±1 sign vector via LCG PRNG.
fn deterministic_signs(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (state >> 63) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Build Randomized Hadamard Transform matrices.
/// R[i][j] = H[i][j] * signs[j],  R^T[i][j] = H[i][j] * signs[i].
fn build_rht(dim: usize) -> (Array, Array) {
    let h = generate_hadamard(dim);
    let signs = deterministic_signs(dim, 42);
    let mut r = vec![0.0f32; dim * dim];
    let mut rt = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            r[i * dim + j] = h[i * dim + j] * signs[j];
            rt[i * dim + j] = h[i * dim + j] * signs[i];
        }
    }
    (
        Array::from_slice(&r, &[dim as i32, dim as i32]),
        Array::from_slice(&rt, &[dim as i32, dim as i32]),
    )
}

/// Lloyd-Max optimal centroids and decision boundaries for N(0,1).
/// After rotating a unit vector with dim d, each coordinate is ~N(0, 1/d).
/// We scale data by sqrt(d) before quantization so these N(0,1) tables apply.
fn lloyd_max_codebook(bits: u8) -> (&'static [f32], &'static [f32]) {
    match bits {
        2 => (
            &[-1.510, -0.453, 0.453, 1.510],
            &[-0.982, 0.0, 0.982],
        ),
        3 => (
            &[-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152],
            &[-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748],
        ),
        4 => (
            &[
                -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
                 0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733,
            ],
            &[
                -2.401, -1.844, -1.437, -1.099, -0.800, -0.522, -0.258, 0.0,
                 0.258,  0.522,  0.800,  1.099,  1.437,  1.844,  2.401,
            ],
        ),
        _ => panic!("TurboQuant supports 2, 3, or 4 bits, got {}", bits),
    }
}

/// Fused quantize→dequantize round-trip for K and V.
/// Applies TurboQuant noise (rotate → bin → codebook lookup → inverse rotate)
/// in a single pass. Returns (keys, values) in bf16.
fn turbo_round_trip_kv(
    keys: &Array,
    values: &Array,
    config: &TurboQuantConfig,
) -> Result<(Array, Array), Exception> {
    // Concat K,V along heads → [B, 2H, T, D], single pipeline
    let kv = mlx_rs::ops::concatenate_axis(&[keys, values], 1)?
        .as_dtype(mlx_rs::Dtype::Float32)?;

    // L2 norm per position → [B, 2H, T, 1]
    let norm = mlx_rs::ops::sqrt(
        &mlx_rs::ops::sum_axis(&(&kv * &kv), -1, Some(true))?,
    )?;

    // Normalize → rotate → scale to N(0,1)
    let kv_scaled = &mlx_rs::ops::matmul(
        &(&kv / &(&norm + &config.eps)),
        &config.rotation,
    )? * &config.sqrt_d;

    // Boundary search → codebook lookup (quantize + dequantize in one shot)
    let indices = mlx_rs::ops::sum_axis(
        &mlx_rs::ops::expand_dims(&kv_scaled, -1)?.ge(&config.boundaries)?,
        -1, Some(false),
    )?.as_dtype(mlx_rs::Dtype::Int32)?;
    let vals = mlx_rs::ops::indexing::take_axis(&config.codebook, &indices, 0)?;

    // Unscale → inverse rotate → rescale by norm → bf16
    let kv_out = (&mlx_rs::ops::matmul(
        &(&vals * &config.inv_sqrt_d),
        &config.rotation_t,
    )? * &norm).as_dtype(mlx_rs::Dtype::Bfloat16)?;

    let parts = mlx_rs::ops::split(&kv_out, 2, Some(1))?;
    Ok((parts[0].clone(), parts[1].clone()))
}

/// Append new array to cached, or return new if no cache yet.
fn concat_or_init(cached: &Option<Array>, new: Array) -> Result<Array, Exception> {
    match cached {
        Some(c) => mlx_rs::ops::concatenate_axis(&[c, &new], 2),
        None => Ok(new),
    }
}

/// Keep the last `window` positions along axis 2. No-op if input is already small enough.
fn tail_along_time(full: &Array, window: usize) -> Result<Array, Exception> {
    let len = full.dim(2) as usize;
    if len <= window {
        return Ok(full.clone());
    }
    let cut = (len - window) as i32;
    let parts = mlx_rs::ops::split_sections(full, &[cut], Some(2))?;
    Ok(parts[1].clone())
}

/// Immutable quantization constants shared across all cache entries.
struct TurboQuantConfig {
    codebook: Array,
    boundaries: Array,
    rotation: Array,
    rotation_t: Array,
    sqrt_d: Array,
    inv_sqrt_d: Array,
    eps: Array,
}

/// KV cache for a single attention layer.
/// Supports both plain bf16 storage and TurboQuant compression.
/// When `window` is Some(W), stored K/V is capped at the last W positions
/// (sliding-window attention); `offset` still advances monotonically so RoPE
/// sees absolute positions.
pub struct KVCache {
    inner: KVCacheInner,
    /// Total tokens seen so far (monotonic). Used as RoPE offset.
    offset: usize,
    /// When Some, cap stored K/V to the last `window` positions after each update.
    window: Option<usize>,
}

enum KVCacheInner {
    Plain {
        keys: Option<Array>,
        values: Option<Array>,
    },
    Quantized {
        keys: Option<Array>,
        values: Option<Array>,
        config: TurboQuantConfig,
    },
}

impl KVCache {
    pub fn new() -> Self {
        Self::new_with_window(None)
    }

    pub fn new_with_window(window: Option<usize>) -> Self {
        Self {
            inner: KVCacheInner::Plain {
                keys: None,
                values: None,
            },
            offset: 0,
            window,
        }
    }

    pub fn new_quantized(head_dim: usize, bits: u8) -> Self {
        Self::new_quantized_with_window(head_dim, bits, None)
    }

    pub fn new_quantized_with_window(head_dim: usize, bits: u8, window: Option<usize>) -> Self {
        let (centroids, bounds) = lloyd_max_codebook(bits);
        let (rotation, rotation_t) = build_rht(head_dim);

        Self {
            inner: KVCacheInner::Quantized {
                keys: None,
                values: None,
                config: TurboQuantConfig {
                    codebook: Array::from_slice(centroids, &[centroids.len() as i32]),
                    boundaries: Array::from_slice(bounds, &[bounds.len() as i32]),
                    rotation,
                    rotation_t,
                    sqrt_d: Array::from_f32((head_dim as f32).sqrt()),
                    inv_sqrt_d: Array::from_f32(1.0 / (head_dim as f32).sqrt()),
                    eps: Array::from_f32(1e-8),
                },
            },
            offset: 0,
            window,
        }
    }

    /// Monotonic position count (for RoPE). Not the stored cache length.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Actual number of positions held in the stored K/V buffer.
    /// Equals `offset()` for unbounded caches; capped at `window` otherwise.
    pub fn stored_len(&self) -> usize {
        let k = match &self.inner {
            KVCacheInner::Plain { keys, .. } => keys.as_ref(),
            KVCacheInner::Quantized { keys, .. } => keys.as_ref(),
        };
        k.map(|a| a.dim(2) as usize).unwrap_or(0)
    }

    /// Read cached K/V without mutating. Returns None if empty.
    /// For speculative attention that must not modify the real cache.
    /// Works with both plain and TurboQuant caches (quantized stores bf16 after round-trip).
    pub fn peek_kv(&self) -> Option<(Array, Array)> {
        match &self.inner {
            KVCacheInner::Plain { keys: Some(k), values: Some(v) } => {
                Some((k.clone(), v.clone()))
            }
            KVCacheInner::Quantized { keys: Some(k), values: Some(v), .. } => {
                Some((k.clone(), v.clone()))
            }
            _ => None,
        }
    }

    /// Store new keys/values and return all cached K/V for SDPA.
    ///
    /// The *returned* (k, v) always include every newly-appended token (so the
    /// current SDPA call sees its own context). The *stored* (k, v) may be
    /// truncated to the last `window` positions when sliding-window attention
    /// is configured, so subsequent calls only pay memory + compute for the
    /// window.
    pub fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
        perf: &PerfStats,
    ) -> Result<(Array, Array), Exception> {
        let new_len = keys.dim(2) as usize;

        match &mut self.inner {
            KVCacheInner::Plain {
                keys: cached_keys,
                values: cached_values,
            } => {
                let k = concat_or_init(cached_keys, keys)?;
                let v = concat_or_init(cached_values, values)?;
                self.offset += new_len;
                let (stored_k, stored_v) = match self.window {
                    Some(w) => (tail_along_time(&k, w)?, tail_along_time(&v, w)?),
                    None => (k.clone(), v.clone()),
                };
                *cached_keys = Some(stored_k);
                *cached_values = Some(stored_v);
                Ok((k, v))
            }

            KVCacheInner::Quantized {
                keys: cached_keys,
                values: cached_values,
                config,
            } => {
                let t = Instant::now();
                let (new_k, new_v) = turbo_round_trip_kv(&keys, &values, config)?;
                mlx_rs::transforms::eval([&new_k, &new_v].into_iter())?;
                perf.acc(&perf.kv_quant_eval, t.elapsed());

                let k = concat_or_init(cached_keys, new_k)?;
                let v = concat_or_init(cached_values, new_v)?;

                self.offset += new_len;
                let (stored_k, stored_v) = match self.window {
                    Some(w) => (tail_along_time(&k, w)?, tail_along_time(&v, w)?),
                    None => (k.clone(), v.clone()),
                };
                *cached_keys = Some(stored_k);
                *cached_values = Some(stored_v);
                Ok((k, v))
            }
        }
    }
}

pub enum Cache {
    KV(KVCache),
}

impl Cache {
    pub fn as_kv_mut(&mut self) -> &mut KVCache {
        match self {
            Cache::KV(kv) => kv,
        }
    }

    pub fn as_kv_ref(&self) -> Option<&KVCache> {
        match self {
            Cache::KV(kv) => Some(kv),
        }
    }

    pub fn kv_offset(&self) -> usize {
        match self {
            Cache::KV(kv) => kv.offset(),
        }
    }

    pub fn kv_stored_len(&self) -> usize {
        match self {
            Cache::KV(kv) => kv.stored_len(),
        }
    }
}
