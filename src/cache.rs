use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lru::LruCache;

use crate::expert_store::{ExpertData, ExpertStore};

/// Thread-safe LRU cache for expert weight data, evicting by byte budget.
pub struct ExpertCache {
    cache: LruCache<(u32, u32), Arc<ExpertData>>,
    max_bytes: usize,
    current_bytes: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ExpertCache {
    pub fn new(max_bytes: usize) -> Self {
        // Use a large cap; actual eviction is by byte budget
        let cap = NonZeroUsize::new(65536).unwrap();
        ExpertCache {
            cache: LruCache::new(cap),
            max_bytes,
            current_bytes: 0,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get cached experts or load missing ones from the store.
    /// Returns experts in the same order as `expert_indices`.
    pub fn get_or_load(
        &mut self,
        store: &ExpertStore,
        layer_idx: u32,
        expert_indices: &[u32],
    ) -> std::io::Result<Vec<Arc<ExpertData>>> {
        let mut results = Vec::with_capacity(expert_indices.len());
        let mut missing_indices = Vec::new();
        let mut missing_positions = Vec::new();

        // Check cache for each expert
        for (pos, &expert_idx) in expert_indices.iter().enumerate() {
            let key = (layer_idx, expert_idx);
            if let Some(data) = self.cache.get(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                results.push(Some(Arc::clone(data)));
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
                results.push(None);
                missing_indices.push(expert_idx);
                missing_positions.push(pos);
            }
        }

        // Batch-load missing experts from SSD
        if !missing_indices.is_empty() {
            let loaded = store.load_experts(layer_idx, &missing_indices)?;
            for (loaded_data, &pos) in loaded.into_iter().zip(missing_positions.iter()) {
                let arc_data = Arc::new(loaded_data);
                let key = (layer_idx, expert_indices[pos]);
                let byte_size = arc_data.byte_size();

                // Evict until we have room
                while self.current_bytes + byte_size > self.max_bytes {
                    if let Some((_, evicted)) = self.cache.pop_lru() {
                        self.current_bytes -= evicted.byte_size();
                    } else {
                        break;
                    }
                }

                self.current_bytes += byte_size;
                self.cache.put(key, Arc::clone(&arc_data));
                results[pos] = Some(arc_data);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    pub fn stats(&self) -> (u64, u64, f64) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        (hits, misses, rate)
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }
}
