use std::collections::HashSet;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap2::Mmap;
use mlx_rs::{Array, Dtype};
use serde_json::Value;

/// Per-tensor offset info parsed from a safetensors header.
struct TensorInfo {
    /// Byte offset of this tensor's data from `data_start`
    data_offset: usize,
    /// Bytes per single expert slice
    per_expert_stride: usize,
    /// Per-expert shape (shape[1:]), e.g. [512, 512] for gate_proj.weight
    expert_shape: Vec<i32>,
    /// MLX dtype
    dtype: Dtype,
}

/// Parsed safetensors layout for one layer file.
struct LayerTensorOffsets {
    /// Start of tensor data: 8 + header_size
    data_start: usize,
    /// The 9 expert tensors in order:
    /// gate_proj.{weight,scales,biases}, up_proj.{weight,scales,biases}, down_proj.{weight,scales,biases}
    tensors: Vec<TensorInfo>,
}

/// The 9 expert arrays extracted for a set of active experts.
/// Each array has shape [num_experts, d1, d2].
pub struct ExpertSlice {
    pub gate_weight: Array,
    pub gate_scales: Array,
    pub gate_biases: Array,
    pub up_weight: Array,
    pub up_scales: Array,
    pub up_biases: Array,
    pub down_weight: Array,
    pub down_scales: Array,
    pub down_biases: Array,
}

/// Manages expert safetensors files with direct pread() extraction.
///
/// Instead of mmap demand-paging (which triggers ~55K page faults/token at ~20μs each),
/// uses pread() syscalls that read entire expert strides in single I/O operations.
/// mmap is kept only for warm set madvise prefetch.
pub struct ExpertMemoryManager {
    files: Vec<File>,       // for pread() extraction
    maps: Vec<Mmap>,        // for warm set madvise only
    offsets: Vec<LayerTensorOffsets>,
    warm_set: HashSet<(u32, u32)>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

fn safetensors_dtype_to_mlx(dtype_str: &str) -> Dtype {
    match dtype_str {
        "U32" => Dtype::Uint32,
        "BF16" => Dtype::Bfloat16,
        "F16" => Dtype::Float16,
        "F32" => Dtype::Float32,
        "I32" => Dtype::Int32,
        "U8" => Dtype::Uint8,
        _ => panic!("unsupported safetensors dtype: {}", dtype_str),
    }
}

/// Parse a safetensors header to extract per-tensor byte offsets, strides, shapes, and dtypes.
fn parse_layer_offsets(mmap: &[u8]) -> anyhow::Result<LayerTensorOffsets> {
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_size;
    let header: Value = serde_json::from_slice(&mmap[8..data_start])?;

    let tensor_names = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
        "down_proj.weight", "down_proj.scales",  "down_proj.biases",
    ];

    let mut tensors = Vec::with_capacity(9);
    for name in &tensor_names {
        let info = header.get(*name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {} in safetensors header", name))?;

        let data_offsets = info["data_offsets"].as_array()
            .ok_or_else(|| anyhow::anyhow!("no data_offsets for {}", name))?;
        let start = data_offsets[0].as_u64().unwrap() as usize;
        let end = data_offsets[1].as_u64().unwrap() as usize;

        let shape: Vec<usize> = info["shape"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let dtype_str = info["dtype"].as_str().unwrap();
        let dtype = safetensors_dtype_to_mlx(dtype_str);
        let num_experts = shape[0]; // always 256
        let expert_shape: Vec<i32> = shape[1..].iter().map(|&s| s as i32).collect();

        let total_bytes = end - start;
        let per_expert_stride = total_bytes / num_experts;

        tensors.push(TensorInfo {
            data_offset: start,
            per_expert_stride,
            expert_shape,
            dtype,
        });
    }

    Ok(LayerTensorOffsets { data_start, tensors })
}

impl ExpertMemoryManager {
    /// Open expert safetensors files: mmap for headers + madvise, File handles for pread.
    pub fn new(expert_dir: &Path, num_layers: usize) -> anyhow::Result<Self> {
        let mut files = Vec::with_capacity(num_layers);
        let mut maps = Vec::with_capacity(num_layers);
        let mut offsets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let path = expert_dir.join(format!("layer_{:02}_experts.safetensors", i));
            let file = File::open(&path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let layer_offsets = parse_layer_offsets(&mmap)?;
            offsets.push(layer_offsets);
            maps.push(mmap);
            // Reopen for pread (separate fd avoids any interaction with mmap)
            files.push(File::open(&path)?);
        }
        Ok(Self {
            files,
            maps,
            offsets,
            warm_set: HashSet::new(),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        })
    }

    /// Record the warm set for hit rate tracking.
    pub fn set_warm_set(&mut self, experts: &[(u32, u32)]) {
        self.warm_set = experts.iter().copied().collect();
    }

    /// Return (hits, misses, hit_rate). Resets counters.
    pub fn take_hit_stats(&self) -> (usize, usize, f64) {
        let h = self.hits.swap(0, Ordering::Relaxed);
        let m = self.misses.swap(0, Ordering::Relaxed);
        let rate = if h + m > 0 { h as f64 / (h + m) as f64 } else { 0.0 };
        (h, m, rate)
    }

    /// Dummy methods for compatibility with engine.rs cache reporting
    pub fn take_cache_stats(&self) -> (u64, u64, f64) { (0, 0, 0.0) }
    pub fn reset_cache_stats(&self) {}
    pub fn cache_size(&self) -> usize { 0 }

    /// Extract specific experts from a layer using pread() syscalls.
    /// Each expert stride is read in a single pread, avoiding the ~20μs-per-page
    /// overhead of mmap demand-paging.
    /// Tracks warm set hit/miss stats.
    pub fn extract_experts(&self, layer: usize, expert_indices: &[i32]) -> ExpertSlice {
        // Track warm set hits
        for &eidx in expert_indices {
            if self.warm_set.contains(&(layer as u32, eidx as u32)) {
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        let file = &self.files[layer];
        let layer_offsets = &self.offsets[layer];
        let n = expert_indices.len() as i32;

        let mut arrays = Vec::with_capacity(9);
        for tensor in &layer_offsets.tensors {
            let stride = tensor.per_expert_stride;
            let total = expert_indices.len() * stride;
            let mut buf = vec![0u8; total];

            for (i, &eidx) in expert_indices.iter().enumerate() {
                let file_offset = (layer_offsets.data_start
                    + tensor.data_offset
                    + eidx as usize * stride) as u64;
                file.read_exact_at(
                    &mut buf[i * stride..(i + 1) * stride],
                    file_offset,
                )
                .expect("pread failed");
            }

            let mut shape = vec![n];
            shape.extend_from_slice(&tensor.expert_shape);
            let arr = unsafe {
                Array::from_raw_data(
                    buf.as_ptr() as *const std::ffi::c_void,
                    &shape,
                    tensor.dtype,
                )
            };
            arrays.push(arr);
        }

        ExpertSlice {
            gate_weight: arrays.remove(0),
            gate_scales: arrays.remove(0),
            gate_biases: arrays.remove(0),
            up_weight: arrays.remove(0),
            up_scales: arrays.remove(0),
            up_biases: arrays.remove(0),
            down_weight: arrays.remove(0),
            down_scales: arrays.remove(0),
            down_biases: arrays.remove(0),
        }
    }

    /// Prefetch warm set expert pages into kernel page cache via madvise.
    /// Returns total bytes advised.
    pub fn mlock_warm_set(&self, experts: &[(u32, u32)]) -> usize {
        let page_size: usize = 16384; // Apple Silicon page size
        let mut advised = 0usize;

        for &(layer, expert_idx) in experts {
            let layer = layer as usize;
            let expert_idx = expert_idx as usize;

            if layer >= self.maps.len() {
                continue;
            }
            let mmap = &self.maps[layer];
            let layer_offsets = &self.offsets[layer];

            for tensor in &layer_offsets.tensors {
                let abs_start = layer_offsets.data_start
                    + tensor.data_offset
                    + expert_idx * tensor.per_expert_stride;
                let len = tensor.per_expert_stride;

                let aligned_start = abs_start & !(page_size - 1);
                let aligned_len = (abs_start + len - aligned_start + page_size - 1)
                    & !(page_size - 1);

                unsafe {
                    let ptr = mmap.as_ptr().add(aligned_start);
                    libc::madvise(ptr as *mut _, aligned_len, libc::MADV_WILLNEED);
                    advised += aligned_len;
                }
            }
        }

        advised
    }
}
