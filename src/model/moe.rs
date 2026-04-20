use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::memory::{ExpertMemoryManager, SingleExpertTensors};
use crate::model::mlp::QuantizedLinear;
use crate::perf::PerfStats;

/// Co-occurrence predictor: predicts next layer's experts from current layer's actual routing.
/// Built from calibration data. At inference, a table lookup replaces the GPU router projection.
pub struct CooccurrencePredictor {
    /// cooccur[layer_pair][expert_i * num_experts + expert_j] = P(j at L+1 | i at L)
    /// layer_pair index = source layer (0..num_layers-1)
    tables: Vec<Vec<f32>>,
    num_experts: usize,
    pred_k: usize,
}

impl CooccurrencePredictor {
    /// Predict top-k experts for `next_layer` given `actual_experts` routed at current layer.
    /// Returns sorted, deduplicated expert indices.
    pub fn predict(&self, layer: usize, actual_experts: &[i32]) -> Vec<i32> {
        if layer >= self.tables.len() {
            return Vec::new();
        }
        let table = &self.tables[layer];
        let n = self.num_experts;

        // Sum co-occurrence rows for each active expert
        let mut scores = vec![0.0f32; n];
        for &e in actual_experts {
            let row_start = e as usize * n;
            for j in 0..n {
                scores[j] += table[row_start + j];
            }
        }

        // Find top-k by partial sort
        let mut indices: Vec<i32> = (0..n as i32).collect();
        let k = self.pred_k.min(n);
        indices.select_nth_unstable_by(k - 1, |&a, &b| {
            scores[b as usize].partial_cmp(&scores[a as usize]).unwrap()
        });
        indices.truncate(k);
        indices.sort();
        indices
    }

    /// Save co-occurrence tables to a binary file.
    /// Format: [num_layers: u32][num_experts: u32][pred_k: u32][tables: f32...]
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(self.tables.len() as u32).to_le_bytes())?;
        f.write_all(&(self.num_experts as u32).to_le_bytes())?;
        f.write_all(&(self.pred_k as u32).to_le_bytes())?;
        for table in &self.tables {
            for &v in table {
                f.write_all(&v.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Load co-occurrence tables from a binary file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let num_layers = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let num_experts = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let pred_k = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let table_size = num_experts * num_experts;
        let mut tables = Vec::with_capacity(num_layers);
        let mut offset = 12;
        for _ in 0..num_layers {
            let mut table = Vec::with_capacity(table_size);
            for _ in 0..table_size {
                table.push(f32::from_le_bytes(data[offset..offset+4].try_into().unwrap()));
                offset += 4;
            }
            tables.push(table);
        }
        Ok(Self { tables, num_experts, pred_k })
    }
}

/// Collects per-layer routing decisions during calibration to build co-occurrence tables.
pub struct CalibrationRecorder {
    num_layers: usize,
    num_experts: usize,
    /// Per-token routing: routing_log[token][layer] = sorted expert indices
    routing_log: Vec<Vec<Vec<i32>>>,
    /// Current token's routing (built layer by layer)
    current_token: Vec<Vec<i32>>,
}

impl CalibrationRecorder {
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            num_layers,
            num_experts,
            routing_log: Vec::new(),
            current_token: Vec::with_capacity(num_layers),
        }
    }

    /// Record which experts were routed at this layer for the current token.
    pub fn record_layer(&mut self, _layer: usize, experts: &[i32]) {
        self.current_token.push(experts.to_vec());
    }

    /// Finalize current token's routing and start a new token.
    pub fn end_token(&mut self) {
        if self.current_token.len() == self.num_layers {
            let token_routing = std::mem::replace(
                &mut self.current_token,
                Vec::with_capacity(self.num_layers),
            );
            self.routing_log.push(token_routing);
        } else {
            self.current_token.clear();
        }
    }

    /// Build co-occurrence predictor from recorded data.
    pub fn build_predictor(&self, pred_k: usize) -> CooccurrencePredictor {
        let n = self.num_experts;
        let num_pairs = self.num_layers - 1;
        let mut tables = vec![vec![0u32; n * n]; num_pairs];

        // Count co-occurrences: for each token, for each layer pair (L, L+1),
        // for each expert_i at L and expert_j at L+1, increment count.
        for token_routing in &self.routing_log {
            for l in 0..num_pairs {
                let experts_l = &token_routing[l];
                let experts_l1 = &token_routing[l + 1];
                for &ei in experts_l {
                    for &ej in experts_l1 {
                        tables[l][ei as usize * n + ej as usize] += 1;
                    }
                }
            }
        }

        // Normalize: each row sums to 1 (conditional probability)
        let float_tables: Vec<Vec<f32>> = tables
            .into_iter()
            .map(|table| {
                let mut ftable = vec![0.0f32; n * n];
                for i in 0..n {
                    let row_start = i * n;
                    let row_sum: u32 = table[row_start..row_start + n].iter().sum();
                    if row_sum > 0 {
                        let inv = 1.0 / row_sum as f32;
                        for j in 0..n {
                            ftable[row_start + j] = table[row_start + j] as f32 * inv;
                        }
                    }
                }
                ftable
            })
            .collect();

        eprintln!(
            "Built co-occurrence predictor: {} layer-pairs, {} experts, {} tokens, pred_k={}",
            num_pairs, n, self.routing_log.len(), pred_k
        );

        CooccurrencePredictor {
            tables: float_tables,
            num_experts: n,
            pred_k,
        }
    }
}


/// Router weights for Level B speculative prediction (pre-converted for CPU execution).
pub struct RouterWeightsRef {
    /// Router scale vector [hidden_size], pre-converted to f32
    pub router_scale_f32: Vec<f32>,
    /// Packed 4-bit weights [num_experts, hidden_size/8]
    pub proj_weight_u32: Vec<u32>,
    /// Dequant scales [num_experts, num_groups], pre-converted to f32
    pub proj_scales_f32: Vec<f32>,
    /// Dequant biases [num_experts, num_groups], pre-converted to f32
    pub proj_biases_f32: Vec<f32>,
    pub num_experts: usize,
    pub hidden_size: usize,
    pub group_size: usize,
    pub root_size: f32,
    pub rms_norm_eps: f32,
}


/// Tracks prediction accuracy and manages co-occurrence prediction / calibration.
pub struct TransitionProfiler {
    num_layers: usize,
    gate_total: usize,
    gate_hits: usize,
    gate_per_layer_total: Vec<usize>,
    gate_per_layer_hits: Vec<usize>,
    /// Pending prediction from previous layer (layer_idx, predicted_experts)
    pub pending_prediction: Option<(usize, Vec<i32>)>,
    /// Co-occurrence predictor (loaded from calibration data)
    pub cooccur: Option<CooccurrencePredictor>,
    /// Calibration recorder (active during --calibrate)
    pub recorder: Option<CalibrationRecorder>,
    /// Post-attention hidden state from current layer (set by forward_gemma4, consumed by MoE)
    pub h_post_attn: Option<Array>,
    /// Router weights per layer for Level B prediction
    pub router_weights: Vec<RouterWeightsRef>,
}

impl TransitionProfiler {
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            gate_total: 0,
            gate_hits: 0,
            gate_per_layer_total: vec![0; num_layers],
            gate_per_layer_hits: vec![0; num_layers],
            pending_prediction: None,
            cooccur: None,
            recorder: None,
            h_post_attn: None,
            router_weights: Vec::new(),
        }
    }

    /// Compare predicted experts against actual routing.
    pub fn record_gate_reuse(&mut self, layer: usize, predicted: &[i32], actual: &[i32]) {
        for &e in actual {
            self.gate_total += 1;
            self.gate_per_layer_total[layer] += 1;
            if predicted.contains(&e) {
                self.gate_hits += 1;
                self.gate_per_layer_hits[layer] += 1;
            }
        }
    }

    /// End of token: clear pending prediction and finalize calibration recording.
    pub fn end_token(&mut self) {
        self.pending_prediction = None;
        if let Some(ref mut rec) = self.recorder {
            rec.end_token();
        }
    }

    /// Report prediction accuracy.
    pub fn report(&self) {
        if self.gate_total > 0 {
            let method = if !self.router_weights.is_empty() {
                "level-B router"
            } else if self.cooccur.is_some() {
                "co-occurrence"
            } else {
                "router-projection"
            };
            let k = if !self.router_weights.is_empty() { 12 } else {
                self.cooccur.as_ref().map(|c| c.pred_k).unwrap_or(12)
            };
            eprintln!("\n=== Prediction ({method}, top-{k}) ===");
            eprintln!(
                "  Overall: {:.1}% ({}/{})",
                self.gate_hits as f64 / self.gate_total as f64 * 100.0,
                self.gate_hits,
                self.gate_total
            );
            for i in 0..self.num_layers {
                let t = self.gate_per_layer_total[i];
                if t > 0 {
                    let h = self.gate_per_layer_hits[i];
                    eprintln!(
                        "  Layer {:>2}: {:.1}% ({}/{})",
                        i,
                        h as f64 / t as f64 * 100.0,
                        h,
                        t
                    );
                }
            }
        }
    }
}

/// If true, use zero-copy mmap + per-expert quantized_matmul instead of gather_qmm.
/// Cached check for NOREACTIVE env var (avoids 40 env::var lookups per token).
static NOREACTIVE: OnceLock<bool> = OnceLock::new();

/// Gemma4 MoE block: different router (norm + scale + per_expert_scale), GELU activation.
/// No shared expert — dense MLP is handled at the layer level.
pub struct Gemma4MoeBlock {
    /// Router projection (hidden_size → num_experts)
    pub router_proj: QuantizedLinear,
    /// Per-dimension learned scale applied after norm
    pub router_scale: Array,
    /// Per-expert output scale
    pub per_expert_scale: Array,
    /// 1/sqrt(hidden_size) for router normalization
    pub root_size: f32,
    pub rms_norm_eps: f32,
    pub top_k: usize,
    pub layer_idx: usize,
    pub bits: i32,
    pub group_size: i32,
}

impl Gemma4MoeBlock {
    pub fn forward(
        &self,
        x: &Array,
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        let k = self.top_k as i32;

        // 1. Router: RMSNormNoScale → scale → proj → softmax → top-k → renormalize → per_expert_scale
        let normed = self.rms_norm_no_scale(x)?;
        let root_size = Array::from_f32(self.root_size).as_dtype(x.dtype())?;
        let scaled = &normed * &root_size;
        let router_scale = self.router_scale.as_dtype(x.dtype())?;
        let scaled = &scaled * &router_scale;
        let expert_scores = self.router_proj.forward(&scaled)?;
        let router_probs = mlx_rs::ops::softmax_axis(&expert_scores, -1, None)?;

        let neg_scores = mlx_rs::ops::negative(&expert_scores)?;
        let inds_full = mlx_rs::ops::argpartition_axis(&neg_scores, k - 1, -1)?;
        let parts = mlx_rs::ops::split_sections(&inds_full, &[k], Some(-1))?;
        let inds = parts[0].as_dtype(mlx_rs::Dtype::Int32)?;

        let scores = mlx_rs::ops::indexing::take_along_axis(&router_probs, &inds, Some(-1))?;
        let s = mlx_rs::ops::sum_axis(&scores, -1, Some(true))?;
        let scores = &scores / &s;
        let flat_idx_for_scale = inds.reshape(&[-1])?;
        let per_exp_scales = mlx_rs::ops::indexing::take_axis(&self.per_expert_scale, &flat_idx_for_scale, 0)?;
        let per_exp_scales = per_exp_scales.reshape(inds.shape())?;
        let per_exp_scales_typed = per_exp_scales.as_dtype(scores.dtype())?;
        let scores = &scores * &per_exp_scales_typed;

        // 2. Compute routing indices
        let flat_idx = inds.reshape(&[-1])?;
        let scores_f32 = scores.as_dtype(mlx_rs::Dtype::Float32)?;

        let seq_len = x.dim(1);
        let is_decode = seq_len == 1;

        // GPU sync: routing + scores (h_post_attn materializes as a dependency)
        let _t = Instant::now();
        mlx_rs::transforms::eval([&flat_idx, &scores_f32])?;
        perf.acc(&perf.moe_routing_eval, _t.elapsed());
        let flat_data: &[i32] = flat_idx.as_slice();
        let scores_data: &[f32] = scores_f32.as_slice();

        // Find unique experts
        let _t = Instant::now();
        let mut unique: Vec<i32> = flat_data.to_vec();
        unique.sort();
        unique.dedup();

        // Reactive prefetch
        if is_decode && !*NOREACTIVE.get_or_init(|| std::env::var("NOREACTIVE").is_ok()) {
            let group = mem.prefetch_gcd_reactive(self.layer_idx, &unique);
            mem.wait_prefetch_group(group);
        }

        // Pre-compute per-expert score weights
        let mut weight_map: HashMap<i32, f32> = HashMap::new();
        for (i, &idx) in flat_data.iter().enumerate() {
            *weight_map.entry(idx).or_insert(0.0) += scores_data[i];
        }

        perf.acc(&perf.routing_cpu, _t.elapsed());

        // Accuracy tracking + prediction for next layer
        if let Some(tp_ref) = tp {
            let mut tp_mut = tp_ref.borrow_mut();

            // Check pending prediction from PREVIOUS layer (must happen before setting new one)
            if let Some((pred_layer, predicted)) = tp_mut.pending_prediction.take() {
                if pred_layer == self.layer_idx {
                    tp_mut.record_gate_reuse(self.layer_idx, &predicted, &unique);
                }
            }

            // Record routing for calibration
            if let Some(ref mut rec) = tp_mut.recorder {
                rec.record_layer(self.layer_idx, &unique);
            }

            // Level A.5 runs in TextModel::forward after this returns.
            // Co-occurrence is a fallback when h_post_attn was not set.
            if is_decode && self.layer_idx + 1 < tp_mut.num_layers
                && tp_mut.h_post_attn.is_none()
            {
                if let Some(ref cooccur) = tp_mut.cooccur {
                    let predicted = cooccur.predict(self.layer_idx, &unique);
                    tp_mut.pending_prediction = Some((self.layer_idx + 1, predicted));
                }
            }
        }

        // Zero-copy expert evaluation with GELU activation
        let _t = Instant::now();
        let experts: Vec<SingleExpertTensors> = {
            let mut v = Vec::with_capacity(unique.len());
            for &eidx in &unique {
                v.push(mem.extract_expert_zerocopy(self.layer_idx, eidx));
            }
            v
        };

        let mut y_accum: Option<Array> = None;

        for (idx, &eidx) in unique.iter().enumerate() {
            let expert = &experts[idx];

            // Expert MLP: GELU(gate_proj(x)) * up_proj(x) → down_proj
            let gate_out = mlx_rs::ops::quantized_matmul(
                x, &expert.gate_weight, &expert.gate_scales, &expert.gate_biases,
                true, self.group_size, self.bits,
            )?;
            let up_out = mlx_rs::ops::quantized_matmul(
                x, &expert.up_weight, &expert.up_scales, &expert.up_biases,
                true, self.group_size, self.bits,
            )?;
            // GELU instead of SiLU
            let act = &mlx_rs::nn::gelu_approximate(&gate_out)? * &up_out;
            let down_out = mlx_rs::ops::quantized_matmul(
                &act, &expert.down_weight, &expert.down_scales, &expert.down_biases,
                true, self.group_size, self.bits,
            )?;

            if is_decode {
                let total_weight = weight_map.get(&eidx).copied().unwrap_or(0.0);
                if total_weight == 0.0 {
                    continue;
                }
                let scale = Array::from_f32(total_weight).as_dtype(x.dtype())?;
                let weighted = &down_out * &scale;
                y_accum = Some(match y_accum {
                    None => weighted,
                    Some(acc) => &acc + &weighted,
                });
            } else {
                let eidx_arr = Array::from_int(eidx);
                let mask = inds.eq(&eidx_arr)?;
                let mask_f = mask.as_dtype(scores.dtype())?;
                let per_pos_weight =
                    mlx_rs::ops::sum_axis(&(&scores * &mask_f), -1, Some(true))?;
                let weighted = &down_out * &per_pos_weight;
                y_accum = Some(match y_accum {
                    None => weighted,
                    Some(acc) => &acc + &weighted,
                });
            }
        }
        perf.acc(&perf.extract_experts, _t.elapsed());

        Ok(y_accum.unwrap_or_else(|| Array::zeros::<f32>(&x.shape()).unwrap()))
    }

    /// RMSNorm without learnable scale (inline for router)
    fn rms_norm_no_scale(&self, x: &Array) -> Result<Array, Exception> {
        let x2 = x * x;
        let mean = mlx_rs::ops::mean_axes(&x2, &[-1], Some(true))?;
        let eps = Array::from_f32(self.rms_norm_eps);
        let rms = mlx_rs::ops::rsqrt(&(&mean + &eps))?;
        Ok(x * &rms)
    }
}
