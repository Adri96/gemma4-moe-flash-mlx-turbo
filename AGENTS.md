# AGENTS.md

## Git

Do NOT add Codex / Claude as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check `memory/MEMORY.md` before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for sparse MoE models on Mac M1/M4 base 16 GB. All-Rust single binary via `mlx-rs`, on-demand expert loading via GCD prefetch + zero-copy mmap Metal buffers.

**Supported model:** **Gemma 4 26B-A4B** at Unsloth UD-MLX-4bit only. Earlier targets (Qwen 3.5 35B-A3B, Gemma 4 UD-3bit) are no longer supported. Some struct names (`QwenTokenizer`, `ExpertLayout::Qwen`, `ExpertFormat::Safetensors`) are vestigial — see `CLAUDE.md` for the full mapping.

## Build

```bash
cargo build --release
```

## Run

```bash
# Split (one-time): converts HF safetensors → resident.safetensors + per-layer ECB files
./target/release/flash-moe split \
  --model-path ./gemma4-ud-4bit \
  --output-path ./split_gemma4_ud4

# Generate
./target/release/flash-moe generate \
  --model-path ./split_gemma4_ud4 \
  --tokenizer-path ./gemma4-ud-4bit \
  --prompt "Hello" --max-tokens 256
```

Optional flags: `--no-speculate`, `--warm-set`, `--kv-quant-bits N`, `--no-kv-quant`, `--stats`, `--debug-tokens`, `--chat`.

## Architecture

### `src/`

-   **main.rs** — CLI (clap): `split` + `generate` subcommands.
-   **config.rs** — `TextModelArgs` (Gemma4) + `QuantizationConfig` with per-component bit-width overrides (Unsloth UD format).
-   **model/** — `Model` / `TextModel` / `DecoderLayer` (Gemma4-only):
    -   `gemma4_attention.rs` — sliding + full attention, K==V for full, scale=1.0, per-layer RoPE.
    -   `moe.rs` — `Gemma4MoeBlock` (router with `RMSNormNoScale` + scale + per-expert scale, GELU). Includes `TransitionProfiler`, Level B / Level C predictors, `CooccurrencePredictor`.
    -   `mlp.rs` — `QuantizedLinear`, `GeLUMLP` (dense MLP runs in parallel with MoE).
    -   `norm.rs` — `RMSNorm`, `RMSNormNoScale`.
-   **memory.rs** — `ExpertMemoryManager`: GCD prefetch (speculative + reactive, with cancellation), zero-copy mmap extraction, warm set pread.
-   **cache.rs** — KV cache with optional TurboQuant (Hadamard rotation + Lloyd-Max codebook, 2/3/4-bit).
-   **engine.rs** — `generate()` loop, nucleus sampling, ghost-token filter for structural markers.
-   **ffi.rs / ffi_zerocopy.cpp** — `array_from_mmap` zero-copy via Metal `newBufferWithBytesNoCopy`.
-   **splitter.rs** — original model → resident + per-layer expert ECB files.
-   **tokenizer.rs** — HF tokenizer + minijinja chat template (struct still named `QwenTokenizer`).
-   **perf.rs** — per-phase timing accumulator.
-   Per-layer eval barriers at CPU/GPU boundaries (argsort is CPU, expert quant matmul is GPU).

### Model

-   Gemma 4 26B-A4B, Unsloth UD-MLX-4bit:
    -   30 layers, 128 experts, top_k=8, hidden_size=2816, MoE intermediate=704, dense MLP intermediate=2112.
    -   24 sliding + 6 full attention layers (K==V on full).
    -   Mixed precision: 4-bit experts + dense MLP + router, 8-bit attention, 6-bit embedding.
    -   ECB file ~428 MB/layer, 8 active = ~26.8 MB/layer (~803 MB I/O per decode token).

## Performance

-   ~3–4 tok/s decode on M4 Mac mini (16 GB) with TurboQuant KV + GCD speculative + reactive prefetch.

## Key gotchas

### MLX / mlx-rs

-   MLX Metal `quantized_matmul` / `gather_qmm` supports only **{2, 3, 4, 6, 8}**-bit. UD models occasionally have 5-bit weights — `load_qlinear_flex()` re-quantizes them at load time.
-   `mlx_array_new_data` and `Array::from_raw_data` **copy** data — use `array_from_mmap` (Metal `newBufferWithBytesNoCopy`) for zero-copy.
-   `Array::load_safetensors()` creates lazy arrays; loading every expert file causes 25+ GB swap on a 16 GB machine.
-   `gather_qmm` is NOT in mlx-rs — use `mlx_sys::mlx_gather_qmm` via FFI wrapper.
-   `argsort` runs on CPU; need eval boundaries before GPU quantized matmul.
-   `bf16 × f32` scalar promotes to f32 — cast scalars to input dtype.

### Memory / UMA

-   **Do NOT load all expert files via `load_safetensors`** — use the on-demand mmap zero-copy path.
-   **`mlock` HURTS**: page table wire/unwire contends with GPU at the kernel `vm_map` level. Use `madvise(WILLNEED)` + prefault touch instead.
-   **`madvise(WILLNEED)` alone is unreliable** — combine with prefault touch (one byte per 16 KB page) inside GCD workers.
-   Per-layer eval frees expert arrays each layer (peak ~26.8 MB, not cumulative).
-   Expert LRU caching does NOT help — working set >> cache size on 16 GB.

### I/O strategy

-   **GCD reactive (default)**: after routing eval determines actual experts, dispatch `F_RDADVISE` + `madvise` + prefault on userInitiated queue. Block via `dispatch_group`. GPU eval runs fault-free.
-   **GCD speculative (default unless `--no-speculate`)**: prefault-only on utility queue, cancellable page-by-page via atomic generation counter. Fires for the next layer's predicted experts after `async_eval`. Reactive cancels it when exact experts are known.
-   **Why prefault-only for speculative**: `F_RDADVISE` and `madvise` issue kernel-level I/O that can't be cancelled — they cause SSD contention with reactive. Prefault touch is the only cancellable warming primitive.

### Known issues

-   Generation visibly stops after ~220 tokens — output stream halts even when the loop continues. Root cause still open; see `memory/` for the running investigation log.
