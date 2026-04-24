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

## 3. Reduce TurboQuant round-trip cost (or drop it)

**Status.** Measured 2026-04-24. The current tree has **no inline eval** in
`update_and_fetch` — the round-trip is already lazy and folds into the next
`async_eval` boundary. The cost is the arithmetic itself, not a sync barrier.

**Measurement (71-token decode, Gemma 4 UD-4bit, M4 base 16 GB).** Instrumented
with per-op `eval` barriers and buckets (`kv_quant_eval`, `sdpa_eval`), then
reverted:

| Metric                | TurboQuant 3-bit | Plain bf16    |
|-----------------------|------------------|---------------|
| KV quant eval (ms/tok)| 18.6             | 0             |
| SDPA eval (ms/tok)    | 13.7             | 22.1          |
| Attention-related sum | **32.3**         | **22.1**      |
| Wall clock (tok/s)    | 5.4              | **5.6**       |

Note: the individual buckets are polluted by MLX lazy-graph attribution —
upstream projections/RoPE fold into whichever bucket evals first. The *sum*
is the clean comparison. TurboQuant adds ~10 ms/tok of real compute (~5% of
a ~180 ms/tok budget). Wall-clock delta ~6 ms/tok confirms it, with some
absorbed by overlap.

**Why it costs anything.** The round-trip is rotate → boundary search →
inverse rotate per new row — two `dim × dim` matmuls (dim = 256 or 512) + a
codebook lookup, across all 30 layers. It's not the eval barrier; it's the ops.

**Second finding.** The "Quantized" cache *stores bf16 after round-trip*, not
packed integer codes. So TurboQuant as-shipped provides **no KV memory win** —
only quantization noise (regularization-like). At 1K context the KV footprint
is already tens of MB; the feature is accuracy-oriented, not memory-oriented.

**Options, in order of effort.**
- **Drop TurboQuant entirely** if the accuracy signal isn't load-bearing.
  One-liner in `main.rs` (flip `--no-kv-quant` to default). Frees ~5% throughput.
- **Pack integer codes in the cache** (a real project): change `Quantized`
  storage to `[N, H, T, ceil(D·bits/8)]` byte arrays, write a custom Metal
  kernel that dequants inside the attention tile. Gets the memory win
  TurboQuant's name implies. Prerequisite: the memory becomes worth caring
  about — i.e. targeting contexts where the KV cache is the bottleneck.
- **Fuse rotate + boundary search into one Metal kernel.** Mid effort. Saves
  most of the ~5%. Worth it only if TurboQuant stays.

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

---

## Resolved via measurement (not pursued)

- **FlashAttention-style tiling for long prefill.** `mlx_rs::fast::scaled_dot_product_attention`
  *is* MLX's Metal flash-attention kernel — tiled QK, online softmax, no
  materialized N×N matrix. Confirmed 2026-04-24 by reading
  `src/model/gemma4_attention.rs:80-92` (call sites) and verifying the fused
  kernel receives bf16 K/V directly (no dequant on read path — the
  `Quantized` cache also stores bf16). Nothing to implement; the win is
  already banked.
