use std::cell::Cell;
use std::time::Duration;

// ── Process memory sampling (macOS) ─────────────────────────────────────────

#[repr(C)]
struct MachTaskBasicInfo {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time: [u32; 2],
    system_time: [u32; 2],
    policy: i32,
    suspend_count: i32,
}

const MACH_TASK_BASIC_INFO: i32 = 20;

extern "C" {
    fn mach_task_self() -> u32;
    fn task_info(
        target: u32,
        flavor: i32,
        info: *mut std::ffi::c_void,
        count: *mut u32,
    ) -> i32;
}

/// Current resident set size (bytes). Darwin-specific; 0 on failure.
/// Darwin's RSS count includes speculative/purgeable pages that are reclaimable.
pub fn current_rss_bytes() -> u64 {
    let mut info = std::mem::MaybeUninit::<MachTaskBasicInfo>::uninit();
    let mut count = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32;
    let rc = unsafe {
        task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as *mut std::ffi::c_void,
            &mut count,
        )
    };
    if rc != 0 { return 0; }
    unsafe { info.assume_init().resident_size }
}

/// Peak resident set size (bytes) for this process. Uses getrusage.
/// macOS reports ru_maxrss in bytes; Linux reports kilobytes.
pub fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0;
        }
        #[cfg(target_os = "macos")]
        { usage.ru_maxrss as u64 }
        #[cfg(not(target_os = "macos"))]
        { usage.ru_maxrss as u64 * 1024 }
    }
}

/// Per-phase timing accumulator for decode performance analysis.
/// All values in nanoseconds. Single-threaded (Cell, not Atomic).
///
/// Per-layer counters accumulate once per decoder layer per token (30× for Gemma 4).
pub struct PerfStats {
    // Eval barriers — where GPU actually syncs
    pub gdn_proj_eval: Cell<u64>,     // vestigial (GDN removed) — always 0
    pub moe_routing_eval: Cell<u64>,  // eval(flat_idx) — routing + attn tail
    pub moe_sort_eval: Cell<u64>,     // vestigial (argsort fused) — always 0
    pub kv_quant_eval: Cell<u64>,     // eval(new_k, new_v) — TurboQuant round-trip
    pub sdpa_eval: Cell<u64>,         // eval(sdpa_out) — fused flash attention
    pub layer_eval: Cell<u64>,        // eval(h) total wall time (residual after sub-phase evals)
    pub eval_wait: Cell<u64>,         // eval(h) after async_eval — page fault time

    // CPU work between evals
    pub extract_experts: Cell<u64>,   // zero-copy extract + per-expert MLP lazy build
    pub routing_cpu: Cell<u64>,       // unique/sort/dedup/HashMap + reactive prefetch wait
}

impl PerfStats {
    pub fn new() -> Self {
        Self {
            gdn_proj_eval: Cell::new(0),
            moe_routing_eval: Cell::new(0),
            moe_sort_eval: Cell::new(0),
            kv_quant_eval: Cell::new(0),
            sdpa_eval: Cell::new(0),
            layer_eval: Cell::new(0),
            eval_wait: Cell::new(0),
            extract_experts: Cell::new(0),
            routing_cpu: Cell::new(0),
        }
    }

    pub fn acc(&self, field: &Cell<u64>, elapsed: Duration) {
        field.set(field.get() + elapsed.as_nanos() as u64);
    }

    pub fn reset(&self) {
        self.gdn_proj_eval.set(0);
        self.moe_routing_eval.set(0);
        self.moe_sort_eval.set(0);
        self.kv_quant_eval.set(0);
        self.sdpa_eval.set(0);
        self.layer_eval.set(0);
        self.eval_wait.set(0);
        self.extract_experts.set(0);
        self.routing_cpu.set(0);
    }

    pub fn report(&self, num_tokens: usize) {
        let ms = |ns: u64| ns as f64 / 1_000_000.0;
        let per_tok = |ns: u64| if num_tokens > 0 { ms(ns) / num_tokens as f64 } else { 0.0 };

        let evals_total = self.gdn_proj_eval.get()
            + self.moe_routing_eval.get()
            + self.moe_sort_eval.get()
            + self.kv_quant_eval.get()
            + self.sdpa_eval.get()
            + self.layer_eval.get();
        let cpu_total = self.extract_experts.get() + self.routing_cpu.get();
        let total = evals_total + cpu_total;

        let pct = |ns: u64| if total > 0 { ns as f64 / total as f64 * 100.0 } else { 0.0 };

        eprintln!("\n=== Perf Breakdown ({} decode tokens) ===", num_tokens);
        eprintln!("Phase                    Total ms   ms/tok    %");
        eprintln!("─────────────────────────────────────────────────");
        if self.gdn_proj_eval.get() > 0 {
            eprintln!("GDN proj eval:         {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.gdn_proj_eval.get()), per_tok(self.gdn_proj_eval.get()), pct(self.gdn_proj_eval.get()));
        }
        eprintln!("MoE routing eval:      {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.moe_routing_eval.get()), per_tok(self.moe_routing_eval.get()), pct(self.moe_routing_eval.get()));
        if self.moe_sort_eval.get() > 0 {
            eprintln!("MoE sort eval:         {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.moe_sort_eval.get()), per_tok(self.moe_sort_eval.get()), pct(self.moe_sort_eval.get()));
        }
        if self.kv_quant_eval.get() > 0 {
            eprintln!("KV quant eval:         {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.kv_quant_eval.get()), per_tok(self.kv_quant_eval.get()), pct(self.kv_quant_eval.get()));
        }
        if self.sdpa_eval.get() > 0 {
            eprintln!("SDPA eval:             {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.sdpa_eval.get()), per_tok(self.sdpa_eval.get()), pct(self.sdpa_eval.get()));
        }
        eprintln!("Layer eval:            {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.layer_eval.get()), per_tok(self.layer_eval.get()), pct(self.layer_eval.get()));
        if self.eval_wait.get() > 0 {
            eprintln!("  └ GPU wait:          {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.eval_wait.get()), per_tok(self.eval_wait.get()), pct(self.eval_wait.get()));
        }
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("  Eval subtotal:       {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(evals_total), per_tok(evals_total), pct(evals_total));
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("Extract experts:       {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.extract_experts.get()), per_tok(self.extract_experts.get()), pct(self.extract_experts.get()));
        eprintln!("Routing CPU:           {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.routing_cpu.get()), per_tok(self.routing_cpu.get()), pct(self.routing_cpu.get()));
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("  CPU subtotal:        {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(cpu_total), per_tok(cpu_total), pct(cpu_total));
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("  ACCOUNTED TOTAL:     {:>8.1}   {:>6.1}",
            ms(total), per_tok(total));
        eprintln!("  Implied tok/s:       {:>8.1}", if per_tok(total) > 0.0 { 1000.0 / per_tok(total) } else { 0.0 });
    }
}
