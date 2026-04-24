use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::cache::KVCache;
use crate::model::mlp::QuantizedLinear;
use crate::model::norm::{RMSNorm, RMSNormNoScale};

/// Gemma4 attention layer (both sliding and full attention).
///
/// Key differences from Qwen attention:
/// - No output gating (no sigmoid gate)
/// - K==V support for full attention layers (V = raw K before k_norm)
/// - V gets RMSNormNoScale (no learnable weight)
/// - scale = 1.0 (not 1/sqrt(d))
/// - Different head_dim and num_kv_heads per layer type
pub struct Gemma4Attention {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: Option<QuantizedLinear>, // None when use_k_eq_v
    pub o_proj: QuantizedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub v_norm: RMSNormNoScale,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: i32,
    pub rope_theta: f32,
    pub use_k_eq_v: bool,
    /// If > 0, at decode (l==1) use a manual softmax(QK)+mask → where(w<τ,0,w) · V path
    /// instead of the fused SDPA, so low-weight V rows contribute 0. 0.0 disables.
    pub sparse_v_threshold: f32,
}

impl Gemma4Attention {
    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let l = x.dim(1);

        // Q projection + reshape + norm
        let queries = self.q_proj.forward(x)?
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?;
        let queries = self.q_norm.forward(&queries)?;

        // K projection + reshape
        let keys = self.k_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?;

        // V: either from v_proj or K==V (raw K before k_norm)
        let values = if self.use_k_eq_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
                .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?
        };

        // K norm (after V capture for K==V)
        let keys = self.k_norm.forward(&keys)?;
        // V norm (no learnable scale)
        let values = self.v_norm.forward(&values)?;

        // Transpose to [B, heads, L, head_dim]
        let keys = mlx_rs::ops::transpose_axes(&keys, &[0, 2, 1, 3])?;
        let values = mlx_rs::ops::transpose_axes(&values, &[0, 2, 1, 3])?;
        let queries = mlx_rs::ops::transpose_axes(&queries, &[0, 2, 1, 3])?;

        // RoPE
        let offset = cache.offset() as i32;
        let queries = mlx_rs::fast::rope(
            &queries, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset, None::<&Array>,
        )?;
        let keys = mlx_rs::fast::rope(
            &keys, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset, None::<&Array>,
        )?;

        // KV cache update
        let (keys, values) = cache.update_and_fetch(keys, values)?;

        // Hybrid path: at decode (l==1) with sparse-V enabled, run a manual
        // softmax(Q @ K^T + mask) → gate-by-threshold → @ V. Prefill keeps the
        // fused SDPA (which is MLX's flash-attention kernel — we'd lose
        // tiled QK and online softmax otherwise, and sparse-V has no benefit
        // there anyway). Gemma 4 uses SDPA scale=1.0 literally.
        let decode_sparse = l == 1 && self.sparse_v_threshold > 0.0;
        let output = if decode_sparse {
            sparse_v_attention(&queries, &keys, &values, mask, self.num_heads, self.num_kv_heads, self.sparse_v_threshold)?
        } else if let Some(m) = mask {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(m),
            )?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?
        };

        // Transpose back: [B, L, num_heads * head_dim]
        let output = mlx_rs::ops::transpose_axes(&output, &[0, 2, 1, 3])?;
        let output = output.reshape(&[b, l, -1])?;

        // Output projection (no gating unlike Qwen)
        self.o_proj.forward(&output)
    }

    /// Speculative attention: runs Q/K/V projections, concatenates with existing
    /// cached K/V (virtual append), runs SDPA, but does NOT modify the cache.
    /// Returns None if cached_kv is None (empty cache during prefill).
    pub fn forward_speculative(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cached_kv: Option<(Array, Array)>,
        offset: usize,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let l = x.dim(1);

        let queries = self.q_proj.forward(x)?
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?;
        let queries = self.q_norm.forward(&queries)?;

        let keys = self.k_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?;
        let values = if self.use_k_eq_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
                .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?
        };
        let keys = self.k_norm.forward(&keys)?;
        let values = self.v_norm.forward(&values)?;

        let keys = mlx_rs::ops::transpose_axes(&keys, &[0, 2, 1, 3])?;
        let values = mlx_rs::ops::transpose_axes(&values, &[0, 2, 1, 3])?;
        let queries = mlx_rs::ops::transpose_axes(&queries, &[0, 2, 1, 3])?;

        // RoPE at the speculative position (same offset as real cache)
        let queries = mlx_rs::fast::rope(
            &queries, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset as i32, None::<&Array>,
        )?;
        let keys = mlx_rs::fast::rope(
            &keys, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset as i32, None::<&Array>,
        )?;

        // Virtual append: concatenate with existing cache (no mutation)
        let (keys, values) = if let Some((ck, cv)) = cached_kv {
            let keys = mlx_rs::ops::concatenate_axis(&[&ck, &keys], 2)?;
            let values = mlx_rs::ops::concatenate_axis(&[&cv, &values], 2)?;
            (keys, values)
        } else {
            (keys, values)
        };

        // SDPA with scale=1.0
        let output = if let Some(m) = mask {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(m),
            )?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?
        };

        let output = mlx_rs::ops::transpose_axes(&output, &[0, 2, 1, 3])?;
        let output = output.reshape(&[b, l, -1])?;
        self.o_proj.forward(&output)
    }
}

/// TurboQuant §5 sparse-V attention for decode (Q is length-1).
///
/// Shapes (decode):
///   queries: [B, num_heads,    1, D]
///   keys:    [B, num_kv_heads, T, D]
///   values:  [B, num_kv_heads, T, D]
///   mask (if Some): additive [1, 1, 1, T]
///
/// GQA via 5-D reshape+broadcast (not `repeat_axis`): copying K/V costs
/// hundreds of MB/tok at 1K+ context on Gemma 4's full-attn layers
/// (num_groups=8), more than the sparsity saves. The fused SDPA we skip
/// here does the same broadcast trick internally.
fn sparse_v_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    mask: Option<&Array>,
    num_heads: usize,
    num_kv_heads: usize,
    threshold: f32,
) -> Result<Array, Exception> {
    let b = queries.dim(0);
    let d = queries.dim(3);
    let hkv = num_kv_heads as i32;
    let hg = (num_heads / num_kv_heads) as i32;

    // Q: [B, H, 1, D] → [B, Hkv, Hg, 1, D]
    let q5 = queries.reshape(&[b, hkv, hg, 1, d])?;
    // K: [B, Hkv, T, D] → [B, Hkv, 1, D, T] (transpose last two, add groups dim)
    let k_t = mlx_rs::ops::transpose_axes(keys, &[0, 1, 3, 2])?; // [B, Hkv, D, T]
    let k5 = mlx_rs::ops::expand_dims(&k_t, 2)?;                 // [B, Hkv, 1, D, T]
    // V: [B, Hkv, T, D] → [B, Hkv, 1, T, D]
    let v5 = mlx_rs::ops::expand_dims(values, 2)?;

    // Q @ K^T via broadcast over Hg → [B, Hkv, Hg, 1, T]
    let mut scores = mlx_rs::ops::matmul(&q5, &k5)?;
    if let Some(m) = mask {
        scores = &scores + m;
    }

    let weights = mlx_rs::ops::softmax_axis(&scores, -1, None)?;

    let tau = Array::from_f32(threshold).as_dtype(weights.dtype())?;
    let keep = weights.ge(&tau)?;
    let zero = Array::from_f32(0.0).as_dtype(weights.dtype())?;
    let weights = mlx_rs::ops::which(&keep, &weights, &zero)?;

    // weights @ V → [B, Hkv, Hg, 1, D]
    let out5 = mlx_rs::ops::matmul(&weights, &v5)?;
    out5.reshape(&[b, num_heads as i32, 1, d])
}
