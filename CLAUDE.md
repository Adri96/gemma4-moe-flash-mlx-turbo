# CLAUDE.md

## Git

Do NOT add Claude as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for sparse MoE models on Mac M1/M4 base 16 GB. All-Rust single binary via `mlx-rs`, on-demand expert loading via GCD prefetch + zero-copy mmap Metal buffers.

**Supported model:**

-   **Gemma 4 26B-A4B** — `gemma4` architecture, 30 layers, 128 experts, top_k=8. Uses **Unsloth UD-MLX-4bit** dynamic quantization (4-bit experts, 8-bit attention, 6-bit embedding, `group_size=64`).

Earlier revisions of this project also targeted Qwen 3.5 35B-A3B and a 3-bit Gemma variant. Both have been dropped; Gemma 4 26B-A4B at UD-4bit is the only supported configuration. Some internal names still reflect the older targets (see "Vestigial names" below).

## Build

```bash
cargo build --release
```

## Run

```bash
# Split model (one-time, creates ECB format)
./target/release/flash-moe split \
  --model-path ./gemma4-ud-4bit \
  --output-path ./split_gemma4_ud4

# Generate — Gemma 4 UD-4bit
./target/release/flash-moe generate \
  --model-path ./split_gemma4_ud4 \
  --tokenizer-path ./gemma4-ud-4bit \
  --prompt "Hello" --max-tokens 256

# Optional flags:
#   --no-speculate    Disable speculative prefetch for predicted experts
#   --warm-set        Load warm set at startup (preads frequent experts into page cache)
#   --kv-quant-bits 3 TurboQuant KV cache (3-bit, default)
#   --no-kv-quant     Disable KV cache quantization (plain bf16)
#   --stats           Print perf breakdown
#   --debug-tokens    Print each generated token ID / text
#   --chat            Interactive multi-turn chat after first prompt
```

**Default `--model-path` / `--tokenizer-path` in `main.rs` is `./split_gemma4`.** The actual split output directory is `./split_gemma4_ud4`, so those defaults usually need to be overridden explicitly.

## Architecture

### `src/`

-   **main.rs** — CLI (clap): `split` + `generate` subcommands. Reads quant config from `config.json`.
-   **config.rs** — `TextModelArgs` (Gemma4). `QuantizationConfig` supports per-component bit-width overrides (Unsloth UD format) via `bits_for()` / `group_size_for()` lookups.
-   **model/** — `Model` / `TextModel` / `DecoderLayer`, all Gemma4-only:
    -   **gemma4_attention.rs** — Gemma4 attention (no gating, K==V for full-attn, v_norm, scale=1.0, per-layer RoPE/head_dim). `forward_speculative()` for Level C: runs attention with virtual KV append (no cache mutation).
    -   **moe.rs** — `Gemma4MoeBlock` (router with `RMSNormNoScale` → scale → proj → per_expert_scale, GELU). `TransitionProfiler` with `RouterWeightsRef` for Level B CPU prediction; `CooccurrencePredictor` / `CalibrationRecorder` for statistical-table fallback.
    -   **mlp.rs** — `QuantizedLinear`, `GeLUMLP` (dense MLP that runs in parallel with the MoE every layer).
    -   **norm.rs** — `RMSNorm`, `RMSNormNoScale` (Gemma4 v_norm / router normalization).
    -   **mod.rs** — `DecoderLayer::forward_gemma4`, `TextModel::forward` with Level C lazy prediction, weight loader.
-   **memory.rs** — `ExpertMemoryManager`: GCD prefetch (speculative/reactive with QoS + cancel), zero-copy mmap extraction, warm set pread, F_RDADVISE. **The splitter writes ECB files; `Safetensors` branch is vestigial fallback logic and not used in practice.**
-   **cache.rs** — `KVCache` with plain bf16 or TurboQuant (Hadamard rotation + Lloyd-Max codebook, 2/3/4-bit).
-   **engine.rs** — `generate()` loop + nucleus sampling + ghost-token filter for structural markers.
-   **perf.rs** — `PerfStats`: per-phase timing accumulator (routing eval, layer eval, GPU wait, extract, routing CPU).
-   **ffi.rs** — `gather_qmm` FFI + `array_from_mmap` zero-copy wrapper.
-   **ffi_zerocopy.cpp** — C++ shim: MLX array from mmap via Metal `newBufferWithBytesNoCopy`.
-   **splitter.rs** — model splitter (original → resident + per-layer expert ECB). Auto-detects three upstream layouts: Qwen-style (switch_mlp), Gemma4 (`experts.gate_up_proj`, fused), Gemma4Ud (`experts.switch_glu`, unfused). Only the Gemma4Ud path is exercised by the supported model; the others are kept so splitting still works for other MLX MoE checkpoints someone may bring.
-   **tokenizer.rs** — `QwenTokenizer` struct (historical name — actually a generic HF tokenizer + minijinja chat template; works with Gemma 4's `tokenizer.json` / `chat_template.jinja`). Exposes `thinking_channel_tokens()` for `<|channel>` / `<channel|>`.
-   **build.rs** — compiles `ffi_zerocopy.cpp` with MLX C++ headers.

### Vestigial names

Some identifiers still use the old model names even though Qwen support has been dropped:

-   `QwenTokenizer` (tokenizer.rs) — works fine with Gemma 4 tokenizers. Renaming it is purely cosmetic and would touch many call sites.
-   `ExpertLayout::Qwen` / `ExpertFormat::Safetensors` (splitter / memory) — dead paths in the hot loop but left in for flexibility when splitting arbitrary MoE checkpoints.
-   Comments in `gemma4_attention.rs` / `mod.rs` that explain a design choice "unlike Qwen" — kept for historical context; do not rely on a Qwen path existing in the live engine.

### I/O strategy (current default, USE_ZEROCOPY=true)

-   **GCD reactive prefetch** (default): After routing eval determines actual experts, `prefetch_gcd_reactive()` dispatches F_RDADVISE + `madvise(WILLNEED)` + prefault touch per expert on GCD userInitiated queue. Blocks via `dispatch_group` until all pages resident. Then mmap zerocopy arrays are created — GPU eval is fault-free. Cancels any in-flight speculative workers first to avoid SSD contention. Gated by `NOREACTIVE=1` env var for A/B testing.
-   **GCD speculative prefetch** (on by default, `--no-speculate` to disable): After `async_eval(h)` submits GPU, `prefetch_gcd_speculative()` fires low-priority (utility QoS) **prefault-only** (no F_RDADVISE/`madvise` — those can't be cancelled once issued) for L+1's predicted experts. Cancellable page-by-page via atomic flag — reactive cancels these when exact experts are known. Pages touched before cancellation remain in page cache.
-   **Warm set pread at startup** (opt-in, `--warm-set`): `mlock_warm_set()` uses parallel pread to guarantee warm expert pages are resident.
-   Per-layer eval via `async_eval` + `eval` separates GPU submission from wait time in perf stats.

### Gemma 4 26B-A4B specifics

-   Architecture: `gemma4` / `gemma4_text`.
-   30 layers, all with MoE + dense MLP **in parallel** (output = dense + expert).
-   Attention: 24 sliding (`head_dim=256`, `kv_heads=8`, `rope_theta=10K`) + 6 full (`global_head_dim=512`, `kv_heads=2`, `rope_theta=1M`, K==V).
-   Hidden size 2816, MoE intermediate 704, dense MLP intermediate 2112.
-   Embedding: 6-bit quantized (UD), scaled by √hidden_size. Tied word embeddings via `quantized_matmul`.
-   MoE: 128 experts, top_k=8, GELU activation. Router: `RMSNormNoScale` → scale → proj → per_expert_scale.
-   Extra norms: `pre/post_feedforward_layernorm`, `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_{1,2}`.
-   Layer scalar: per-layer learned scalar applied at end of decoder layer.
-   Logit softcapping: `tanh(logits/30) × 30`.
-   **UD-4bit (current, `group_size=64`)**: Mixed precision — 4-bit experts (3.35 MB each), 8-bit attention, 4-bit dense MLP / router, 6-bit embedding. ECB file ~428 MB/layer, 8 active = ~26.8 MB/layer, ~803 MB I/O per decode token across all 30 layers.
-   Weight naming: UD uses `scales` / `biases`, the mlx-community 4-bit build uses `weight_scales` / `weight_biases`. Both handled by `load_qlinear_flex()`.
-   Source: `unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit`.
-   Default split output dir: `./split_gemma4_ud4` (~15 GB on disk).

### GCD speculative + cancel

-   **Prefault-only for speculative**: speculative workers do ONLY prefault touch (one byte per 16 KB page). `F_RDADVISE` and `madvise` are skipped for speculative because they issue kernel-level I/O that **cannot be cancelled** — causing SSD contention with reactive. Prefault touch is cancellable page-by-page via atomic generation counter.
-   **Reactive uses full pipeline**: `F_RDADVISE` + `madvise(WILLNEED)` + prefault. Only speculative is restricted.
-   Pages touched by speculative before cancellation remain in page cache, reducing reactive's work.
-   GCD QoS (utility vs userInitiated) provides OS-level thread priority differentiation.
-   **Gemma 4 UD-4bit with speculative + TurboQuant KV**: ~3–4 tok/s @256 tokens on M4 base 16 GB.

### Why pread-based speculative failed

-   Experts are ~3.35 MB (small) — page cache retains them between tokens.
-   GPU eval is ~0.65 ms/layer — too short for blocking speculative I/O to overlap.
-   Blocking pread for speculative can't be cancelled — always contends with reactive.
-   Background rayon threads are 2.3× slower than main-thread on macOS.

## Key gotchas

### MLX / mlx-rs

-   **MLX Metal `quantized_matmul`/`gather_qmm` supports only {2, 3, 4, 6, 8}-bit**. Unsloth UD sometimes uses 5-bit for a handful of weights (historically only a few on layer 29); `load_qlinear_flex()` auto-detects unsupported bit widths and re-quantizes at load time (CPU dequantize → GPU quantize to nearest supported bit width).
-   `mlx_array_new_data` (and `Array::from_raw_data`) **copies** data — use `ffi_zerocopy.cpp` shim for zero-copy via `newBufferWithBytesNoCopy`.
-   `Array::load_safetensors()` creates lazy arrays; loading all expert files causes swap storms on 16 GB.
-   `gather_qmm` is NOT in mlx-rs — use `mlx_sys::mlx_gather_qmm` via FFI wrapper.
-   `argsort` runs on CPU; eval boundaries needed before GPU `gather_qmm`.
-   Activation dtype drift: `bf16 × f32` scalar promotes to f32 — cast scalars to input dtype.
-   `array_from_mmap` (Metal buffer creation via `newBufferWithBytesNoCopy`) is fast (0.3 ms for 108 arrays). NOT a bottleneck.

### Memory / UMA

-   **Do NOT load all expert files via `load_safetensors`** — causes swap storms on 16 GB.
-   On-demand expert extraction via mmap + zero-copy is the correct approach.
-   **pread() is 3.6× faster than mmap demand-paging** (page fault overhead: ~107 μs/page for cold 16 KB random reads).
-   **GCD reactive prefault before eval is the current strategy**: `F_RDADVISE` + `madvise(WILLNEED)` + prefault touch (one byte per 16 KB page) on GCD userInitiated queue. Blocks via `dispatch_group`. Equivalent to pread for page warming, with cancellation support.
-   **pread is also effective** (historical default): contiguous MB reads warm page cache; subsequent mmap zerocopy eval runs fault-free. Used at startup for warm set.
-   **`madvise(MADV_WILLNEED)` alone is unreliable** — returns before pages are loaded. Combined with prefault touch in GCD workers, it becomes reliable.
-   **`mlock` HURTS**: page table wire/unwire contends with GPU at kernel `vm_map` level.
-   **Pread-based speculative is a net negative** — can't be cancelled, always contends.
-   **GCD speculative with cancellation works**: fire-and-forget prefault-only on utility queue, cancel via atomic when reactive starts. `F_RDADVISE`/`madvise` skipped for speculative (uncancellable kernel I/O causes contention). No SSD contention.
-   Per-layer eval ensures expert arrays are freed after each layer (peak ~26.8 MB for 4-bit, not cumulative).
-   Expert LRU caching does NOT help — working set >> cache size on 16 GB.
-   **Warm set pread at startup** (opt-in via `--warm-set`): guarantees a significant fraction of expert pages are resident. Less impactful now that GCD speculative provides similar warming.

### I/O architecture findings:

-   **Page faults as flow control**: GPU self-throttles to match SSD throughput. Natural pipelining — but explicit prefetch is more efficient per byte.
-   **Pread-based speculative during eval causes SSD contention**: can't be cancelled. Consistently worse in all tested configurations.
-   **GCD speculative with cancellation avoids contention**: atomic cancel flag lets reactive interrupt speculative. Pages already touched remain resident.
-   **Background par_iter is ~2.3× slower than main-thread par_iter**. GCD dispatch avoids this penalty.

### Env vars for testing:

-   `NOREACTIVE=1` — skip reactive blocking pread (for A/B testing).

### Known issues

-   **Generation stalls after ~220 tokens**: output stops appearing on stdout even though the loop keeps running. Multiple past debug sessions attributed this variously to the ghost-zone filter in `engine.rs`, KV cache growth + TurboQuant, and EOS handling. As of the most recent pass the root cause is still open — see `memory/` for the running investigation log.
