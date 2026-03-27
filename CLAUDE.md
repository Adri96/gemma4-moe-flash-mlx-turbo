# CLAUDE.md

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for Qwen3.5-35B-A3B (36.3 GB, 9-bit MLX) on Mac M4 base 16GB. Streams MoE expert weights from SSD, pins resident weights in RAM, with optional TurboQuant KV cache compression.

## Build

```bash
source .venv/bin/activate  # Python 3.14 via Homebrew
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

## Run

```bash
# Split model (one-time)
python run.py split --model-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit --output-path ./split_model

# Generate
python run.py generate --model-path ./split_model \
  --tokenizer-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit \
  --prompt "Hello" --max-tokens 256 --cache-size-mb 6144

# With TurboQuant KV cache (2, 3, or 4 bit)
python run.py generate ... --kv-bits 4
```

## Architecture

- **Rust (PyO3)**: `src/` — SSD expert loading (`pread` + `F_NOCACHE`), LRU cache, model splitter
- **Python (MLX)**: `flash_qwen/` — model definition, inference engine, TurboQuant KV cache
- Model type is `qwen3_5_moe` mapping to `mlx_lm.models.qwen3_5` (NOT `qwen3_next`)
- 40 layers: 30 linear-attention (GatedDeltaNet/ArraysCache) + 10 full-attention (Attention/KVCache), every 4th layer is full-attention
- Weights are already sanitized (mlx-sanitized: 0.30.7) — do NOT re-sanitize

## Key gotchas

- `mx.linalg.qr` requires `stream=mx.cpu` — not supported on GPU
- MLX has no `searchsorted` — use boundary comparison: `(x[..., None] > boundaries).sum(-1)`
- TurboQuantCache must NOT have a `bits` attribute — triggers quantized SDPA path in `mlx_lm/models/base.py:117`
- Weight prefix: split model has `model.layers.N...`, must re-add `language_model.` at load time
- Bottleneck is SSD expert loading (~59% hit rate cold), not KV cache or compute
