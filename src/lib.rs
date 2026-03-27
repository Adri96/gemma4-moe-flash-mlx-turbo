mod cache;
mod expert_store;
mod splitter;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use cache::ExpertCache;
use expert_store::ExpertStore;

#[pyclass]
struct FlashExpertManager {
    store: ExpertStore,
    cache: Mutex<ExpertCache>,
}

#[pymethods]
impl FlashExpertManager {
    #[new]
    fn new(expert_dir: &str, cache_size_mb: usize) -> PyResult<Self> {
        let store = ExpertStore::new(Path::new(expert_dir))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let cache = Mutex::new(ExpertCache::new(cache_size_mb * 1024 * 1024));
        Ok(FlashExpertManager { store, cache })
    }

    /// Load experts for a given layer.
    ///
    /// Returns a dict with keys like "gate_weight", "gate_scales", "gate_biases",
    /// "up_weight", "up_scales", "up_biases", "down_weight", "down_scales", "down_biases".
    /// Each value is bytes containing the stacked tensor data for the requested experts
    /// in the order specified by expert_indices.
    ///
    /// Also returns shape info under keys like "gate_weight_shape", etc.
    fn load_experts<'py>(
        &self,
        py: Python<'py>,
        layer_idx: u32,
        expert_indices: Vec<u32>,
    ) -> PyResult<HashMap<String, PyObject>> {
        let mut cache = self.cache.lock().unwrap();
        let experts = cache
            .get_or_load(&self.store, layer_idx, &expert_indices)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let header = self.store.header(layer_idx);
        let n = experts.len();

        // Component extraction helper
        struct Component {
            name: &'static str,
            offset: u64,
            size: u64,
            shape: [u32; 2],
        }

        let components = [
            Component {
                name: "gate_weight",
                offset: header.gate_proj.weight_offset,
                size: header.gate_proj.weight_size,
                shape: header.gate_weight_shape,
            },
            Component {
                name: "gate_scales",
                offset: header.gate_proj.scales_offset,
                size: header.gate_proj.scales_size,
                shape: header.gate_scales_shape,
            },
            Component {
                name: "gate_biases",
                offset: header.gate_proj.biases_offset,
                size: header.gate_proj.biases_size,
                shape: header.gate_scales_shape, // same shape as scales
            },
            Component {
                name: "up_weight",
                offset: header.up_proj.weight_offset,
                size: header.up_proj.weight_size,
                shape: header.up_weight_shape,
            },
            Component {
                name: "up_scales",
                offset: header.up_proj.scales_offset,
                size: header.up_proj.scales_size,
                shape: header.up_scales_shape,
            },
            Component {
                name: "up_biases",
                offset: header.up_proj.biases_offset,
                size: header.up_proj.biases_size,
                shape: header.up_scales_shape,
            },
            Component {
                name: "down_weight",
                offset: header.down_proj.weight_offset,
                size: header.down_proj.weight_size,
                shape: header.down_weight_shape,
            },
            Component {
                name: "down_scales",
                offset: header.down_proj.scales_offset,
                size: header.down_proj.scales_size,
                shape: header.down_scales_shape,
            },
            Component {
                name: "down_biases",
                offset: header.down_proj.biases_offset,
                size: header.down_proj.biases_size,
                shape: header.down_scales_shape,
            },
        ];

        let mut result: HashMap<String, PyObject> = HashMap::new();

        for comp in &components {
            let per_expert = comp.size as usize;
            let mut stacked = Vec::with_capacity(n * per_expert);

            for expert in &experts {
                let start = comp.offset as usize;
                let end = start + per_expert;
                stacked.extend_from_slice(&expert.data[start..end]);
            }

            let bytes = PyBytes::new(py, &stacked);
            result.insert(comp.name.to_string(), bytes.into_any().unbind());

            // Shape: [n_experts, dim1, dim2]
            let shape_key = format!("{}_shape", comp.name);
            let shape = (n as u32, comp.shape[0], comp.shape[1]);
            result.insert(shape_key, shape.into_pyobject(py)?.into_any().unbind());
        }

        // Add quant info
        result.insert(
            "quant_bits".to_string(),
            header.quant_bits.into_pyobject(py)?.into_any().unbind(),
        );
        result.insert(
            "quant_group_size".to_string(),
            header.quant_group_size.into_pyobject(py)?.into_any().unbind(),
        );

        Ok(result)
    }

    /// Return (hits, misses, hit_rate) cache statistics.
    fn cache_stats(&self) -> (u64, u64, f64) {
        let cache = self.cache.lock().unwrap();
        cache.stats()
    }

    /// Return current cache size in bytes.
    fn cache_bytes(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.current_bytes()
    }

    /// Return number of layers.
    fn num_layers(&self) -> u32 {
        self.store.num_layers()
    }
}

#[pyfunction]
fn split_model(model_path: &str, output_path: &str) -> PyResult<()> {
    splitter::split_model(Path::new(model_path), Path::new(output_path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlashExpertManager>()?;
    m.add_function(wrap_pyfunction!(split_model, m)?)?;
    Ok(())
}
