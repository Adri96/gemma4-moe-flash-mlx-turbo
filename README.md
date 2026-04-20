# flash-moe

All-Rust inference engine for sparse Mixture-of-Experts models on Apple Silicon. Runs Gemma 4 26B-A4B on an M4 Mac mini with 16 GB of RAM by loading experts on-demand from SSD.

**Supported model:** **Gemma 4 26B-A4B** — 26B parameters, 4B active. 30 layers, 128 experts, top-8. Uses Unsloth's UD-MLX-4bit dynamic quantization (4-bit experts, 8-bit attention, 6-bit embedding).

The key idea: sparse MoE models only activate a small fraction of parameters per token (8 experts out of 128 per layer), so the full model doesn't need to fit in memory. flash-moe keeps resident weights in Metal buffers and loads experts on-demand from SSD via memory-mapped I/O, using GCD-dispatched prefetch to keep pages ahead of the GPU.

## How it works

The model is split into two parts:

- **Resident weights** (~3.0 GB): embeddings, attention, norms, router, dense MLP, output head. Loaded once into Metal buffers at startup.
- **Expert files** (~428 MB/layer): one ECB (expert-centric binary) file per layer. Memory-mapped but never fully loaded — only the 8 active experts are paged in per layer per token.

### I/O pipeline

The bottleneck isn't compute — it's getting expert bytes from SSD to GPU before it stalls. Without explicit prefetch, the GPU triggers page faults that pull data in 16 KB chunks — synchronous kernel traps that reduce effective throughput to a fraction of what sequential reads achieve. flash-moe avoids this with a two-stage GCD prefetch pipeline:

1. **Speculative** (during GPU eval): After submitting the current layer to the GPU, fire off low-priority (utility QoS) GCD workers to prefault pages for the *next* layer's predicted experts. Workers do prefault touch only (no `F_RDADVISE` / `madvise` — those issue kernel-level I/O that can't be cancelled).
2. **Reactive** (after routing): Once the router picks the actual 8 experts, cancel any in-flight speculative work (atomic flag — cancellable page-by-page), then dispatch high-priority (userInitiated QoS) workers with the full I/O pipeline (`F_RDADVISE` + `madvise` + prefault). Blocks until all pages are resident.
3. **Eval** (zero faults): GPU reads from Metal buffers backed by already-resident mmap pages. Pure compute, no page faults.

Cancellation is what makes this work — without it, speculative I/O contends with reactive and throughput drops significantly.

### Expert prediction

Speculative prefetch needs to predict which experts the next layer will select. flash-moe uses **model-based prediction** rather than statistical tables:

- **Level C** (default): Run the current layer's dense MLP to approximate its output (without waiting for MoE), then run the next layer's attention (virtual KV append — no cache mutation) and router on GPU. All lazy ops, eval'd between `async_eval` and `eval`. ~73% top-12 accuracy, GPU wait drops from 30 ms to ~1 ms.
- **Level B** (CPU fallback): Run next layer's router projection on `h_post_attn` via CPU dequantized matmul. Pre-converted f32 weights, zero GPU impact. ~62% accuracy.
- **Co-occurrence tables** (statistical fallback, opt-in via `--calibrate`): Lookup table built from a calibration run. ~50% accuracy.

### Per-token I/O

| Quant | Expert size | Active/layer | Layers | I/O per token |
|-------|-------------|--------------|--------|---------------|
| UD-4bit (mixed) | 3.35 MB | 26.8 MB | 30 | ~803 MB |

Unsloth's dynamic 4-bit quantization uses mixed precision (4-bit experts, 8-bit attention, 6-bit embedding) for better output quality than uniform 4-bit, with no ghost-token artifacts from quantization. Decode rate on a 16 GB M4 Mac mini is ~3–4 tok/s.

### Why not just load everything into RAM?

The 4-bit checkpoint is ~15 GB on disk. On a 16 GB machine that means swap, and swap means page faults during GPU eval — exactly what this project avoids. The MoE sparsity (~6% active) makes on-demand loading viable: you only need the data you're actually using.

## Requirements

- **macOS** on Apple Silicon (tested on M4 Mac mini, 16 GB)
- **Rust** toolchain (stable)
- **Model weights**: [unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit](https://huggingface.co/unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit) (~15 GB)

## Build

```bash
cargo build --release
```

## Usage

### 1. Download the model

```bash
huggingface-cli download unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit \
  --local-dir ./gemma4-ud-4bit
```

### 2. Split the model

Converts HuggingFace safetensors into resident weights + per-layer expert ECB files:

```bash
./target/release/flash-moe split \
  --model-path ./gemma4-ud-4bit \
  --output-path ./split_gemma4_ud4
```

One-time step. You can delete the original download after splitting.

### 3. Generate

```bash
./target/release/flash-moe generate \
  --model-path ./split_gemma4_ud4 \
  --tokenizer-path ./gemma4-ud-4bit \
  --prompt "Explain the Riemann hypothesis in simple terms" \
  --max-tokens 256
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--no-speculate` | off | Disable speculative prefetch for predicted experts |
| `--warm-set` | off | Pread frequent experts into page cache at startup |
| `--kv-quant-bits N` | 3 | TurboQuant KV cache: 2, 3, or 4-bit quantization |
| `--no-kv-quant` | off | Disable KV cache quantization (plain bf16) |
| `--stats` | off | Print per-phase perf breakdown after generation |
| `--debug-tokens` | off | Print every generated token id + decoded text |
| `--chat` | off | After the first prompt, accept follow-up turns from stdin |
| `--calibrate N` | — | Record routing decisions over N tokens, save co-occurrence predictor |

## Known issues

- **Generation visibly stalls after ~220 tokens**: the loop keeps running but no further output appears on stdout. Several causes are plausible — the structural-marker "ghost zone" filter in `engine.rs`, KV cache growth interacting with TurboQuant, EOS handling, or token-level decoder behavior. Investigation is ongoing.

## License

MIT
