# Future optimizations

Pending optimizations for this codebase. Each entry is a self-contained brief so
a future session can pick it up without re-deriving context. Entries are
ordered roughly by expected impact / cost ratio.

Already landed (reference, not to redo):
- TurboQuant KV cache (3-bit, Hadamard + Lloyd-Max) — `src/cache.rs`
- Sliding-window attention mask + per-layer offsets — `src/model/mod.rs`
- Expert speculation (Level B CPU + Level C GPU + co-occurrence) — `src/model/moe.rs`, `src/memory.rs`
- Zero-copy mmap via `newBufferWithBytesNoCopy` — `src/ffi_zerocopy.cpp`
- Unsloth UD mixed-precision quantization — `src/config.rs`
- **Sliding-window KV truncation** (this session) — `src/cache.rs`, `src/model/mod.rs`

---

## 1. Sparse-V attention gating (TurboQuant §5)

**What.** At decode time, zero out V rows whose attention weight falls below a
threshold before the SDPA `softmax(QK) · V` gather. The TurboQuant paper
reports +22.8% decode speed at 32K with Δppl ≈ 0.000 on Gemma 4.

**Why it stacks with what's in the tree.** The 3-bit quantized V stays in
cache; sparse-V only skips the gather for positions the row would have
contributed negligibly to. Orthogonal axis from bit-width compression.

**Where.** `src/model/gemma4_attention.rs::forward`. The `mlx_rs::fast::scaled_dot_product_attention`
call hides QK/softmax/gather behind one op — we lose the ability to threshold
inside it. Two options:
- (a) Drop the fused SDPA for a manual `softmax(QK/scale + mask) · V` so we can
  threshold between softmax and gather.
- (b) Pre-compute an attention-weight mask in a separate pass, then call SDPA
  with V zeroed at low-attention rows. Extra pass but keeps the fused kernel.

**Catch.** The per-row threshold needs to be small enough to preserve quality
(paper uses ~1e-4). Easy to break quality if set too aggressively — validate on
a held-out prompt set.

**Expected gain.** +15–25% decode at long contexts. Negligible at short context.

---

## 2. Token-level speculative decoding + MoE-Spec verification

**What.** A small draft model (e.g. Gemma 3 270M, dense) proposes k tokens; the
main model verifies in parallel. On MoE this normally regresses because the
tree-of-drafts activates many unique experts, blowing memory and hiding
speedup. MoE-Spec (arXiv Feb 2026) fixes this by bounding the expert set during
verification: only load the top-N most-contributive experts per layer, drop the
long tail.

**Why.** The single biggest available win — 10–30% over EAGLE-3 per MoE-Spec,
and speculative decoding itself is a 2–3× baseline win on dense models.
Without it, per-token cost is floored by the sequential expert-load latency.

**Where.**
- New: draft-model runner (could reuse `mlx-rs` with a smaller checkpoint).
- `src/engine.rs::generate` — rework the token loop to verify batches.
- `src/model/moe.rs::Gemma4MoeBlock::forward` — accept a verification-time
  expert budget; reuse the existing router-top-k path but cap total unique
  experts per layer.
- `src/memory.rs::ExpertMemoryManager` — already supports the prefetch pattern,
  needs to take a "budgeted expert set" instead of per-token top-k.

**Dependencies.** Needs Gemma 3 270M tokenizer compatibility with Gemma 4's
tokenizer (vocabularies differ — confirm before investing). If incompatible,
distill a small draft instead, or use shallow-layer-drafting (early-exit from
the main model as draft).

**Risk.** Biggest architectural change in this list. Changes the hot loop.

**Expected gain.** 2–3× tok/s at short context, probably less at long context
since we're SSD-bound there.

---

## 3. Overlap `kv_quant_eval` with next-layer GPU submit

**What.** `KVCache::update_and_fetch` evals the TurboQuant round-trip
synchronously before returning. Visible as `perf.kv_quant_eval` (~18 ms/tok at
1299 tokens, 7.5% of total). The eval happens inline; no reason it can't
overlap with the next layer's async GPU submit.

**Where.** `src/cache.rs::update_and_fetch` — remove the inline
`mlx_rs::transforms::eval` in the Quantized branch, let the lazy graph reach
the next `async_eval` boundary. Verify perf stats still split cleanly.

**Catch.** The `eval` is there because the quantized K/V is then concatenated
with the cached K/V. Concat of a lazy op isn't wrong; the eval just forces
materialization for timing attribution. Dropping it means KV-quant time folds
into the next layer's eval. That's fine for total latency but noisier stats.

**Expected gain.** Up to ~7% of current decode time, likely less due to
dependency chains.

---

## 4. Chunked prefill

**What.** Currently `TextModel::forward` processes the whole prompt in one
pass. Long prompts (2K+) materialize full-size Q/K/V tensors per layer, causing
memory spikes on a 16 GB machine.

**Where.** `src/model/mod.rs::TextModel::forward` and
`src/engine.rs::generate`. Split the prompt into chunks of `window` tokens (or
fewer), forward each, accumulate KV state. Sliding-window truncation already
handles the cache correctly across chunks.

**Catch.** The Level-C speculative prediction path is only useful for
decode, not prefill — make sure the speculative branch is gated off during
chunked prefill to avoid wasted work.

**Expected gain.** Primarily a memory win (avoids OOM on long prompts).
Latency-neutral or slightly slower per token. Prerequisite for long-context
chat.

---

## 5. Persisted co-occurrence table

**What.** `CooccurrencePredictor` is rebuilt per session (see
`src/model/moe.rs::CooccurrencePredictor`). Calibration once on a diverse
prompt set, dump to disk, load at startup → Level-A-accuracy prediction from
token 1 instead of after warmup.

**Where.**
- Add a `--calibrate <corpus>` CLI subcommand to run the existing
  `CalibrationRecorder` against a prompt file and dump the table.
- Add `--cooccur-table <path>` to `generate`/`serve` that loads it into
  `TransitionProfiler`.

**Format.** Flat binary: per layer-pair, a sparse list of (current_expert,
next_experts_top_N). A few MB total.

**Expected gain.** First ~50 tokens of every new session benefit. Small
absolute win, but trivial to implement.

---

## Dismissed / not pursued

**Cross-layer shared KV cache.** The user's research notes framed this as a
Gemma 4 feature; it isn't. Gemma 4's KV savings come from (a) the
sliding/global layer split and (b) global layers' K==V with
`num_global_key_value_heads=2`. Both already exploited. Don't chase it.

**ANE (Orion project) direct execution.** Interesting paper result, but the
M-series ANE is bandwidth-limited for large matrix ops and our bottleneck is
SSD expert I/O, not compute. Orion's 170 tok/s is a 124M dense model — doesn't
generalize. Radar only.

---

## Not yet measured

Some potential wins that need a profile first before committing to
implementation:

- **Prefault reduction for repeat-decode experts.** The GCD reactive prefault
  currently runs unconditionally. If the page-cache-hit rate is high for
  recently-used experts, the prefault is wasted work. Add a bloom filter of
  "pages touched in last N tokens" to skip prefault when hit.
- **KV compression beyond 3-bit** via quantile-compander or QJL for the first
  few prompt tokens (which are higher-variance).
- **FlashAttention-style tiling for long prefill** if `mlx_rs::fast::scaled_dot_product_attention`
  doesn't already do it (haven't checked the kernel).
