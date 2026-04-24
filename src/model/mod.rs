pub mod gemma4_attention;
pub mod mlp;
pub mod moe;
pub mod norm;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::cache::{Cache, KVCache};
use crate::config::TextModelArgs;
use crate::memory::ExpertMemoryManager;
use crate::perf::PerfStats;
use gemma4_attention::Gemma4Attention;
use std::cell::RefCell;
use mlp::{QuantizedLinear, GeLUMLP};
use moe::{Gemma4MoeBlock, TransitionProfiler};
use norm::RMSNorm;

pub enum AttentionLayer {
    Gemma4(Gemma4Attention),
}

pub struct DecoderLayer {
    pub attention: AttentionLayer,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: MoeVariant,
    // Gemma4 extra norms
    pub pre_feedforward_layernorm: Option<RMSNorm>,
    pub post_feedforward_layernorm: Option<RMSNorm>,
    pub post_feedforward_layernorm_1: Option<RMSNorm>,
    pub post_feedforward_layernorm_2: Option<RMSNorm>,
    pub pre_feedforward_layernorm_2: Option<RMSNorm>,
    // Gemma4 dense MLP (runs in parallel with MoE)
    pub dense_mlp: Option<GeLUMLP>,
    // Gemma4 layer scalar
    pub layer_scalar: Option<Array>,
}

pub enum MoeVariant {
    Gemma4(Gemma4MoeBlock),
}

impl DecoderLayer {
    pub fn kv_head_dim(&self) -> usize {
        match &self.attention {
            AttentionLayer::Gemma4(a) => a.head_dim,
        }
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        self.forward_gemma4(x, mask, cache, mem, perf, tp)
    }

    fn forward_gemma4(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        // Attention block
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = match &mut self.attention {
            AttentionLayer::Gemma4(attn) => attn.forward(&h, mask, cache.as_kv_mut())?,
        };
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = &residual + &h;

        // Store h_post_attn (bf16) for Level B prediction — will be read as raw bytes after routing eval
        if let Some(tp_ref) = tp {
            if !tp_ref.borrow().router_weights.is_empty() {
                tp_ref.borrow_mut().h_post_attn = Some(h.clone());
            }
        }

        // Feedforward block: dense MLP + MoE in parallel
        let residual = h.clone();

        // Dense MLP path
        let h1 = self.pre_feedforward_layernorm.as_ref().unwrap().forward(&h)?;
        let h1 = self.dense_mlp.as_ref().unwrap().forward(&h1)?;
        let h1 = self.post_feedforward_layernorm_1.as_ref().unwrap().forward(&h1)?;

        // MoE expert path
        let h2_input = self.pre_feedforward_layernorm_2.as_ref().unwrap().forward(&h)?;
        let h2 = match &self.mlp {
            MoeVariant::Gemma4(moe) => moe.forward(&h2_input, mem, perf, tp)?,
        };
        let h2 = self.post_feedforward_layernorm_2.as_ref().unwrap().forward(&h2)?;

        // Sum dense + expert
        let h = &h1 + &h2;

        let h = self.post_feedforward_layernorm.as_ref().unwrap().forward(&h)?;
        let mut h = &residual + &h;

        // Layer scalar
        if let Some(ref scalar) = self.layer_scalar {
            h = &h * scalar;
        }

        Ok(h)
    }
}

pub struct TextModel {
    pub embed_tokens_weight: Array,
    pub embed_tokens_scales: Option<Array>,
    pub embed_tokens_biases: Option<Array>,
    pub embed_bits: i32,
    pub embed_group_size: i32,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub embed_scale: Option<f32>,
    pub sliding_window: Option<usize>,
}

impl TextModel {
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: &mut [Cache],
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        speculate: bool,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        let flat_ids = input_ids.flatten(None, None)?;
        let h = if let Some(ref scales) = self.embed_tokens_scales {
            let w = mlx_rs::ops::indexing::take_axis(&self.embed_tokens_weight, &flat_ids, 0)?;
            let s = mlx_rs::ops::indexing::take_axis(scales, &flat_ids, 0)?;
            let b = mlx_rs::ops::indexing::take_axis(self.embed_tokens_biases.as_ref().unwrap(), &flat_ids, 0)?;
            mlx_rs::ops::dequantize(&w, &s, &b, Some(self.embed_group_size), Some(self.embed_bits))?
        } else {
            mlx_rs::ops::indexing::take_axis(&self.embed_tokens_weight, &flat_ids, 0)?
        };
        let shape = input_ids.shape();
        let h = h.reshape(&[shape[0], shape[1], -1])?;
        let hidden = if let Some(scale) = self.embed_scale {
            let s = Array::from_f32(scale).as_dtype(h.dtype())?;
            &h * &s
        } else {
            h
        };

        // Create attention masks. Sliding layers cap stored K/V at `sliding_window`,
        // so the mask's column count is `stored_len + seq_len`, not `offset + seq_len`.
        let mut full_offset = 0usize;
        let mut sliding_stored = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if matches!(&layer.attention, AttentionLayer::Gemma4(a) if !a.use_k_eq_v) {
                sliding_stored = cache[i].kv_stored_len();
                break;
            }
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if matches!(&layer.attention, AttentionLayer::Gemma4(a) if a.use_k_eq_v) {
                full_offset = cache[i].kv_offset();
                break;
            }
        }
        let full_mask = create_attention_mask(&hidden, full_offset)?;
        let sliding_mask = create_sliding_mask(&hidden, sliding_stored, self.sliding_window)?;

        let mut h = hidden;
        let num_layers = self.layers.len();

        for i in 0..num_layers {
            let mask = if matches!(&self.layers[i].attention, AttentionLayer::Gemma4(a) if a.use_k_eq_v) {
                full_mask.as_ref()
            } else {
                sliding_mask.as_ref().or(full_mask.as_ref())
            };
            h = self.layers[i].forward(&h, mask, &mut cache[i], mem, perf, tp)?;
            // layers[i] mutable borrow released — can now access layers immutably

            let _t = Instant::now();

            // Level C GPU prediction: dense MLP + next attention + next router (lazy)
            let lazy_pred = if speculate && i + 1 < num_layers {
                let h_pa = tp.and_then(|r| r.borrow_mut().h_post_attn.take());
                if let Some(h_pa) = h_pa {
                    let next_mask = match &self.layers[i + 1].attention {
                        AttentionLayer::Gemma4(a) if a.use_k_eq_v => full_mask.as_ref(),
                        _ => sliding_mask.as_ref().or(full_mask.as_ref()),
                    };
                    Self::predict_level_c(
                        &self.layers[i], &self.layers[i + 1],
                        &h_pa, &cache[i + 1], next_mask,
                    ).ok().flatten()
                } else {
                    None
                }
            } else {
                None
            };

            mlx_rs::transforms::async_eval(std::iter::once(&h))?;

            // Speculative prefetch
            if speculate && i + 1 < num_layers {
                if let Some(ref pred_arr) = lazy_pred {
                    // Level A.5: eval the small prediction (dense MLP + router, independent of expert MLP)
                    mlx_rs::transforms::eval(std::iter::once(pred_arr))?;
                    let pred_data: &[i32] = pred_arr.as_slice();
                    let mut predicted: Vec<i32> = pred_data.to_vec();
                    predicted.sort();
                    predicted.dedup();
                    // Store for accuracy tracking (consumed by next layer's MoE block)
                    if let Some(tp_ref) = tp {
                        tp_ref.borrow_mut().pending_prediction = Some((i + 1, predicted.clone()));
                    }
                    mem.prefetch_gcd_speculative(i + 1, &predicted);
                } else {
                    // Fallback: Level B (CPU) prediction from MoE block
                    if let Some(tp_ref) = tp {
                        let tp_borrow = tp_ref.borrow();
                        if let Some((pred_layer, ref predicted)) = tp_borrow.pending_prediction {
                            if pred_layer == i + 1 {
                                mem.prefetch_gcd_speculative(pred_layer, predicted);
                            }
                        }
                    }
                }
            }

            let _tw = Instant::now();
            mlx_rs::transforms::eval(std::iter::once(&h))?;
            perf.acc(&perf.eval_wait, _tw.elapsed());

            // GPU eval done: safe to release LRU-evicted expert pages from page cache
            mem.flush_evictions(i);

            perf.acc(&perf.layer_eval, _t.elapsed());
        }

        self.norm.forward(&h)
    }

    /// Level C prediction: dense MLP + next-layer attention + next-layer router.
    /// Falls back to Level A.5 (skip attention) if KV cache is quantized or empty.
    /// All computations are lazy GPU ops — caller must eval() the result.
    fn predict_level_c(
        layer_l: &DecoderLayer,
        layer_l1: &DecoderLayer,
        h_post_attn: &Array,
        cache_l1: &Cache,
        mask: Option<&Array>,
    ) -> Result<Option<Array>, Exception> {
        // Dense MLP path (layer L's resident weights) → approximate layer L output
        let h1 = layer_l.pre_feedforward_layernorm.as_ref().unwrap().forward(h_post_attn)?;
        let h1 = layer_l.dense_mlp.as_ref().unwrap().forward(&h1)?;
        let h1 = layer_l.post_feedforward_layernorm_1.as_ref().unwrap().forward(&h1)?;
        let h_approx = layer_l.post_feedforward_layernorm.as_ref().unwrap().forward(&h1)?;
        let mut h_approx = h_post_attn + &h_approx;
        if let Some(ref scalar) = layer_l.layer_scalar {
            h_approx = &h_approx * scalar;
        }

        // Layer L+1's attention on h_approx (speculative, no cache mutation)
        let AttentionLayer::Gemma4(ref attn) = layer_l1.attention;
        let router_input = if let Some(kv_cache) = cache_l1.as_kv_ref() {
            if let Some(cached_kv) = kv_cache.peek_kv() {
                let normed = layer_l1.input_layernorm.forward(&h_approx)?;
                let attn_out = attn.forward_speculative(
                    &normed, mask, Some(cached_kv), kv_cache.offset(),
                )?;
                let attn_out = layer_l1.post_attention_layernorm.forward(&attn_out)?;
                &h_approx + &attn_out
            } else {
                h_approx
            }
        } else {
            h_approx
        };

        // Layer L+1's router
        let MoeVariant::Gemma4(ref moe) = layer_l1.mlp;
        {
            let x2 = &router_input * &router_input;
            let mean = mlx_rs::ops::mean_axes(&x2, &[-1], Some(true))?;
            let eps = Array::from_f32(moe.rms_norm_eps);
            let rms = mlx_rs::ops::rsqrt(&(&mean + &eps))?;
            let normed = &router_input * &rms;
            let root = Array::from_f32(moe.root_size).as_dtype(h_post_attn.dtype())?;
            let scaled = &normed * &root;
            let rs = moe.router_scale.as_dtype(h_post_attn.dtype())?;
            let scaled = &scaled * &rs;
            let scores = moe.router_proj.forward(&scaled)?;

            let neg = mlx_rs::ops::negative(&scores)?;
            let top = mlx_rs::ops::argpartition_axis(&neg, 11, -1)?;
            let parts = mlx_rs::ops::split_sections(&top, &[12i32], Some(-1))?;
            let predicted = parts[0].as_dtype(mlx_rs::Dtype::Int32)?.reshape(&[-1])?;
            Ok(Some(predicted))
        }
    }
}

pub struct Model {
    pub model: TextModel,
    pub lm_head: QuantizedLinear,
    pub tie_word_embeddings: bool,
    pub head_dim: usize,
    pub final_logit_softcapping: Option<f32>,
}

impl Model {
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: &mut [Cache],
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        speculate: bool,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        let out = self.model.forward(input_ids, cache, mem, perf, speculate, tp)?;
        let logits = if self.tie_word_embeddings {
            if self.model.embed_tokens_scales.is_some() {
                mlx_rs::ops::quantized_matmul(
                    &out,
                    &self.model.embed_tokens_weight,
                    self.model.embed_tokens_scales.as_ref().unwrap(),
                    self.model.embed_tokens_biases.as_ref().unwrap(),
                    Some(true),
                    Some(self.model.embed_group_size),
                    Some(self.model.embed_bits),
                )?
            } else {
                let w_t = mlx_rs::ops::transpose_axes(&self.model.embed_tokens_weight, &[1, 0])?.as_dtype(out.dtype())?;
                mlx_rs::ops::matmul(&out, &w_t)?
            }
        } else {
            self.lm_head.forward(&out)?
        };

        // Logit softcapping (Gemma4)
        if let Some(softcap) = self.final_logit_softcapping {
            let s = Array::from_f32(softcap);
            let scaled = &logits / &s;
            let capped = mlx_rs::ops::tanh(&scaled)?;
            Ok(&capped * &s)
        } else {
            Ok(logits)
        }
    }

    /// Enable TurboQuant §5 sparse-V attention gating at decode.
    /// `threshold = 0.0` disables it (default). Typical value: 1e-4.
    pub fn set_sparse_v_threshold(&mut self, threshold: f32) {
        for layer in self.model.layers.iter_mut() {
            match &mut layer.attention {
                AttentionLayer::Gemma4(a) => a.sparse_v_threshold = threshold,
            }
        }
    }

    pub fn make_cache(&self, kv_quant_bits: Option<u8>) -> Vec<Cache> {
        let sliding_window = self.model.sliding_window;
        self.model
            .layers
            .iter()
            .map(|layer| {
                // Only local (non-K==V) attention layers use the sliding window.
                let window = match &layer.attention {
                    AttentionLayer::Gemma4(a) if !a.use_k_eq_v => sliding_window,
                    _ => None,
                };
                match kv_quant_bits {
                    Some(bits) => Cache::KV(KVCache::new_quantized_with_window(
                        layer.kv_head_dim(),
                        bits,
                        window,
                    )),
                    None => Cache::KV(KVCache::new_with_window(window)),
                }
            })
            .collect()
    }
}

// --- Weight loading ---

pub fn load_model(split_path: &Path, args: &TextModelArgs, quant: Option<&crate::config::QuantizationConfig>, verbose: bool) -> anyhow::Result<Model> {
    load_gemma4_model(split_path, args, quant, verbose)
}

fn load_gemma4_model(split_path: &Path, args: &TextModelArgs, quant: Option<&crate::config::QuantizationConfig>, verbose: bool) -> anyhow::Result<Model> {
    if verbose { eprintln!("Loading resident weights (Gemma4)..."); }
    let resident_path = split_path.join("resident/resident.safetensors");
    let weights = load_safetensors_map(&resident_path)?;

    if verbose {
        eprintln!(
            "Loaded {} resident tensors ({:.2} GB)",
            weights.len(),
            weights.values().map(|a| a.nbytes()).sum::<usize>() as f64 / 1e9
        );
    }

    let default_bits = quant.map(|q| q.bits as i32).unwrap_or(8);
    let default_gs = quant.map(|q| q.group_size as i32).unwrap_or(32);
    let has_overrides = quant.map(|q| !q.overrides.is_empty()).unwrap_or(false);

    // Per-component quantization lookup (config paths use original naming)
    let q_bits = |component: &str| -> (i32, i32) {
        match quant {
            Some(q) => (q.bits_for(component), q.group_size_for(component)),
            None => (default_bits, default_gs),
        }
    };

    if verbose {
        if has_overrides {
            eprintln!("  Quantization: mixed-precision (default {}-bit, {} overrides)",
                default_bits, quant.unwrap().overrides.len());
        } else {
            eprintln!("  Quantization: {}-bit, group_size={}", default_bits, default_gs);
        }
    }

    let layer_types = args.layer_types.as_ref().unwrap();
    let global_head_dim = args.global_head_dim.unwrap_or(args.head_dim);
    let num_global_kv_heads = args.num_global_key_value_heads.unwrap_or(args.num_key_value_heads);

    let mut layers = Vec::with_capacity(args.num_hidden_layers);
    for i in 0..args.num_hidden_layers {
        let prefix = format!("language_model.layers.{}", i);
        // Config override paths use original naming: language_model.model.layers.N...
        let cfg_prefix = format!("language_model.model.layers.{}", i);
        let is_full = layer_types[i] == "full_attention";
        let use_k_eq_v = args.attention_k_eq_v && is_full;

        let input_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.input_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_attn_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_attention_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };

        // Attention — look up bits per component (typically 8-bit for UD models)
        let p = format!("{}.self_attn", prefix);
        let (attn_bits, attn_gs) = q_bits(&format!("{}.self_attn.q_proj", cfg_prefix));
        let head_dim = if is_full { global_head_dim } else { args.head_dim };
        let num_kv_heads = if use_k_eq_v { num_global_kv_heads } else { args.num_key_value_heads };
        let (rope_dims, rope_theta) = args.gemma4_rope_config(is_full);

        let v_proj = if use_k_eq_v {
            None
        } else {
            let (vb, vgs) = q_bits(&format!("{}.self_attn.v_proj", cfg_prefix));
            Some(load_qlinear_flex(&weights, &format!("{}.v_proj", p), vb, vgs))
        };

        let attention = AttentionLayer::Gemma4(Gemma4Attention {
            q_proj: load_qlinear_flex(&weights, &format!("{}.q_proj", p), attn_bits, attn_gs),
            k_proj: load_qlinear_flex(&weights, &format!("{}.k_proj", p), attn_bits, attn_gs),
            v_proj,
            o_proj: load_qlinear_flex(&weights, &format!("{}.o_proj", p), attn_bits, attn_gs),
            q_norm: RMSNorm {
                weight: get_weight(&weights, &format!("{}.q_norm.weight", p)),
                eps: args.rms_norm_eps,
            },
            k_norm: RMSNorm {
                weight: get_weight(&weights, &format!("{}.k_norm.weight", p)),
                eps: args.rms_norm_eps,
            },
            v_norm: norm::RMSNormNoScale { eps: args.rms_norm_eps },
            num_heads: args.num_attention_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            rope_theta,
            use_k_eq_v,
            sparse_v_threshold: 0.0,
        });

        // Feedforward norms
        let pre_ffn_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.pre_feedforward_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_ffn_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_feedforward_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_ffn_ln_1 = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_feedforward_layernorm_1.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_ffn_ln_2 = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_feedforward_layernorm_2.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let pre_ffn_ln_2 = RMSNorm {
            weight: get_weight(&weights, &format!("{}.pre_feedforward_layernorm_2.weight", prefix)),
            eps: args.rms_norm_eps,
        };

        let mlp_prefix = format!("{}.mlp", prefix);
        let (mlp_bits, mlp_gs) = q_bits(&format!("{}.mlp.gate_proj", cfg_prefix));
        let dense_mlp = GeLUMLP {
            gate_proj: load_qlinear_flex(&weights, &format!("{}.gate_proj", mlp_prefix), mlp_bits, mlp_gs),
            up_proj: load_qlinear_flex(&weights, &format!("{}.up_proj", mlp_prefix), mlp_bits, mlp_gs),
            down_proj: load_qlinear_flex(&weights, &format!("{}.down_proj", mlp_prefix), mlp_bits, mlp_gs),
        };

        // MoE router
        let router_prefix = format!("{}.router", prefix);
        let (router_bits, router_gs) = q_bits(&format!("{}.router.proj", cfg_prefix));
        // Expert bits (used at inference for gather_qmm)
        let (expert_bits, expert_gs) = q_bits(&format!("{}.experts.switch_glu.gate_proj", cfg_prefix));
        let moe = MoeVariant::Gemma4(Gemma4MoeBlock {
            router_proj: load_qlinear_flex(&weights, &format!("{}.proj", router_prefix), router_bits, router_gs),
            router_scale: get_weight(&weights, &format!("{}.scale", router_prefix)),
            per_expert_scale: get_weight(&weights, &format!("{}.per_expert_scale", router_prefix)),
            root_size: (args.hidden_size as f32).powf(-0.5),
            rms_norm_eps: args.rms_norm_eps,
            top_k: args.experts_per_tok(),
            layer_idx: i,
            bits: expert_bits,
            group_size: expert_gs,
        });

        // Layer scalar
        let layer_scalar = weights.get(&format!("{}.layer_scalar", prefix)).cloned();

        layers.push(DecoderLayer {
            attention,
            input_layernorm: input_ln,
            post_attention_layernorm: post_attn_ln,
            mlp: moe,
            pre_feedforward_layernorm: Some(pre_ffn_ln),
            post_feedforward_layernorm: Some(post_ffn_ln),
            post_feedforward_layernorm_1: Some(post_ffn_ln_1),
            post_feedforward_layernorm_2: Some(post_ffn_ln_2),
            pre_feedforward_layernorm_2: Some(pre_ffn_ln_2),
            dense_mlp: Some(dense_mlp),
            layer_scalar,
        });

        if verbose && ((i + 1) % 10 == 0 || i == args.num_hidden_layers - 1) {
            eprintln!("  Built layer {}/{}", i + 1, args.num_hidden_layers);
        }
    }

    let final_norm = RMSNorm {
        weight: get_weight(&weights, "language_model.norm.weight"),
        eps: args.rms_norm_eps,
    };

    // Embedding — may be quantized (UD models have 6-bit) or plain bf16
    let embed_weight = get_weight(&weights, "language_model.embed_tokens.weight");
    let embed_scales = weights.get("language_model.embed_tokens.scales")
        .or_else(|| weights.get("language_model.embed_tokens.weight_scales"))
        .cloned();
    let embed_biases = weights.get("language_model.embed_tokens.biases")
        .or_else(|| weights.get("language_model.embed_tokens.weight_biases"))
        .cloned();
    let (embed_bits, embed_gs) = q_bits("language_model.model.embed_tokens");

    if verbose {
        if embed_scales.is_some() {
            eprintln!("  Embedding: {}-bit quantized", embed_bits);
        } else {
            eprintln!("  Embedding: bf16 (unquantized)");
        }
    }

    // Dummy lm_head (not used when tie_word_embeddings=true)
    let dummy_lm_head = QuantizedLinear {
        weight: Array::from_f32(0.0),
        scales: Array::from_f32(0.0),
        biases: Array::from_f32(0.0),
        bits: default_bits,
        group_size: default_gs,
    };

    Ok(Model {
        model: TextModel {
            embed_tokens_weight: embed_weight,
            embed_tokens_scales: embed_scales,
            embed_tokens_biases: embed_biases,
            embed_bits,
            embed_group_size: embed_gs,
            layers,
            norm: final_norm,
            embed_scale: Some((args.hidden_size as f32).sqrt()),
            sliding_window: args.sliding_window,
        },
        lm_head: dummy_lm_head,
        tie_word_embeddings: args.tie_word_embeddings,
        head_dim: args.head_dim,
        final_logit_softcapping: args.final_logit_softcapping,
    })
}

// --- Helpers ---

fn load_safetensors_map(path: &Path) -> anyhow::Result<HashMap<String, Array>> {
    let map = Array::load_safetensors(path)
        .map_err(|e| anyhow::anyhow!("failed to load {}: {:?}", path.display(), e))?;
    Ok(map)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Array {
    weights
        .get(key)
        .unwrap_or_else(|| panic!("missing weight: {}", key))
        .clone()
}


/// MLX Metal kernels only support these bit widths for quantized_matmul.
const SUPPORTED_QUANT_BITS: &[i32] = &[2, 3, 4, 6, 8];

/// Find the largest supported bit width <= requested bits.
fn nearest_supported_bits(bits: i32) -> i32 {
    *SUPPORTED_QUANT_BITS
        .iter()
        .rev()
        .find(|&&b| b <= bits)
        .unwrap_or(&4)
}

/// Load quantized linear with flexible naming (Qwen: .scales/.biases, Gemma4: .weight_scales/.weight_biases).
/// If bits is unsupported by MLX Metal kernels (e.g. 5-bit), dequantizes and re-quantizes
/// to the nearest supported bit width at load time.
fn load_qlinear_flex(
    weights: &HashMap<String, Array>,
    prefix: &str,
    bits: i32,
    group_size: i32,
) -> QuantizedLinear {
    let weight = get_weight(weights, &format!("{}.weight", prefix));
    let scales = weights.get(&format!("{}.scales", prefix))
        .or_else(|| weights.get(&format!("{}.weight_scales", prefix)))
        .unwrap_or_else(|| panic!("missing scales for: {}", prefix))
        .clone();
    let biases = weights.get(&format!("{}.biases", prefix))
        .or_else(|| weights.get(&format!("{}.weight_biases", prefix)))
        .unwrap_or_else(|| panic!("missing biases for: {}", prefix))
        .clone();

    if !SUPPORTED_QUANT_BITS.contains(&bits) {
        let target_bits = nearest_supported_bits(bits);
        eprintln!("    Re-quantizing {}: {}-bit → {}-bit (unsupported kernel)", prefix, bits, target_bits);
        // Force dequantize on CPU — Metal has no kernel for this bit width
        let cpu_stream = mlx_rs::Stream::cpu();
        let dequant = mlx_rs::ops::dequantize_device(
            &weight, &scales, &biases, Some(group_size), Some(bits), &cpu_stream,
        ).expect("dequantize failed");
        mlx_rs::transforms::eval(std::iter::once(&dequant)).expect("eval dequantize failed");
        let (new_w, new_s, new_b) = mlx_rs::ops::quantize(&dequant, Some(group_size), Some(target_bits))
            .expect("re-quantize failed");
        mlx_rs::transforms::eval([&new_w, &new_s, &new_b].into_iter()).expect("eval quantize failed");
        QuantizedLinear { weight: new_w, scales: new_s, biases: new_b, bits: target_bits, group_size }
    } else {
        QuantizedLinear { weight, scales, biases, bits, group_size }
    }
}

fn create_attention_mask(
    hidden: &Array,
    cache_offset: usize,
) -> Result<Option<Array>, Exception> {
    let seq_len = hidden.dim(1) as usize;
    if seq_len <= 1 {
        return Ok(None);
    }
    let total_len = cache_offset + seq_len;
    let rows = Array::from_iter(
        (cache_offset as i32)..(total_len as i32),
        &[seq_len as i32, 1],
    );
    let cols = Array::from_iter(0..(total_len as i32), &[1, total_len as i32]);
    let mask = rows.ge(&cols)?;
    let zero = Array::from_f32(0.0);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let additive = mlx_rs::ops::r#where(&mask, &zero, &neg_inf)?;
    let additive = additive.reshape(&[1, 1, seq_len as i32, total_len as i32])?;
    let additive = additive.as_dtype(hidden.dtype())?;
    Ok(Some(additive))
}

/// Sliding-window causal mask over a (possibly truncated) KV buffer.
///
/// `stored_len` is the number of already-cached K positions *before* appending
/// the current `seq_len` tokens, so the mask has `stored_len + seq_len` columns
/// matching the K returned by `KVCache::update_and_fetch`. Row `i` corresponds
/// to the new token at relative position `stored_len + i`; col `j` to relative
/// position `j`. The mask allows `j <= stored_len + i` (causal) and
/// `(stored_len + i) - j < window` (sliding).
///
/// When `window` is None, falls back to a plain causal mask (delegating to
/// `create_attention_mask`, whose `cache_offset` parameter has the same meaning
/// as `stored_len` for unbounded caches).
fn create_sliding_mask(
    hidden: &Array,
    stored_len: usize,
    window: Option<usize>,
) -> Result<Option<Array>, Exception> {
    let seq_len = hidden.dim(1) as usize;
    if seq_len == 0 {
        return Ok(None);
    }
    let Some(window) = window else {
        return create_attention_mask(hidden, stored_len);
    };
    let total_len = stored_len + seq_len;
    // Single-token decode with every stored col in-window: no mask needed.
    if seq_len == 1 && total_len <= window {
        return Ok(None);
    }

    let rows = Array::from_iter(
        (stored_len as i32)..(total_len as i32),
        &[seq_len as i32, 1],
    );
    let cols = Array::from_iter(0..(total_len as i32), &[1, total_len as i32]);
    // Causal: j <= i (relative)
    let causal = rows.ge(&cols)?;
    // Window: i - j < window  ⇔  j > i - window
    let window_arr = Array::from_int(window as i32);
    let lower = &rows - &window_arr;
    let in_window = cols.gt(&lower)?;
    let mask = causal.logical_and(&in_window)?;
    let zero = Array::from_f32(0.0);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let additive = mlx_rs::ops::r#where(&mask, &zero, &neg_inf)?;
    let additive = additive.reshape(&[1, 1, seq_len as i32, total_len as i32])?;
    let additive = additive.as_dtype(hidden.dtype())?;
    Ok(Some(additive))
}
