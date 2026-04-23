use std::cell::RefCell;
use std::io::Write;
use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::memory::ExpertMemoryManager;
use crate::model::Model;
use crate::model::moe::{TransitionProfiler, CooccurrencePredictor, CalibrationRecorder, RouterWeightsRef};
use crate::model::MoeVariant;
use crate::perf::PerfStats;
use crate::tokenizer::QwenTokenizer;

pub fn generate(
    model: &mut Model,
    tokenizer: &QwenTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    rep_penalty: f32,
    mem: &ExpertMemoryManager,
    kv_quant_bits: Option<u8>,
    speculate: bool,
    cooccur: Option<CooccurrencePredictor>,
    recorder: Option<CalibrationRecorder>,
    verbose: bool,
    debug_tokens: bool,
    mut on_token: Option<&mut dyn FnMut(&str)>,
) -> anyhow::Result<(String, Option<CalibrationRecorder>)> {
    let perf = PerfStats::new();
    let num_layers = model.model.layers.len();
    let mut tp_inner = TransitionProfiler::new(num_layers);
    tp_inner.cooccur = cooccur;
    tp_inner.recorder = recorder;

    // Populate Level B router weights from Gemma4 layers (pre-convert to f32 for CPU)
    for layer in &model.model.layers {
        let MoeVariant::Gemma4(ref moe) = layer.mlp;
        let rs_f32 = moe.router_scale.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_rs::transforms::eval(std::iter::once(&rs_f32)).unwrap();
        let rs_data: &[f32] = rs_f32.as_slice();
        let w_data: &[u32] = moe.router_proj.weight.as_slice();
        let s_f32 = moe.router_proj.scales.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        let b_f32 = moe.router_proj.biases.as_dtype(mlx_rs::Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&s_f32, &b_f32]).unwrap();
        let s_data: &[f32] = s_f32.as_slice();
        let b_data: &[f32] = b_f32.as_slice();
        let hidden_size = rs_data.len();
        let num_experts = moe.router_proj.weight.dim(0) as usize;
        tp_inner.router_weights.push(RouterWeightsRef {
            router_scale_f32: rs_data.to_vec(),
            proj_weight_u32: w_data.to_vec(),
            proj_scales_f32: s_data.to_vec(),
            proj_biases_f32: b_data.to_vec(),
            num_experts,
            hidden_size,
            group_size: moe.group_size as usize,
            root_size: moe.root_size,
            rms_norm_eps: moe.rms_norm_eps,
        });
    }
    if verbose && !tp_inner.router_weights.is_empty() {
        eprintln!("Level B prediction (CPU): {} layers", tp_inner.router_weights.len());
    }

    let tp = RefCell::new(tp_inner);
    let input_ids = tokenizer.encode(prompt)?;
    let mut cache = model.make_cache(kv_quant_bits);

    // Prefill
    if verbose { eprintln!("Prefilling {} tokens...", input_ids.len()); }
    let t0 = Instant::now();
    let input = Array::from_slice(
        &input_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        &[1, input_ids.len() as i32],
    );
    let logits = model.forward(&input, &mut cache, mem, &perf, false, None)?;
    mlx_rs::transforms::eval(std::iter::once(&logits))?;
    let prefill_time = t0.elapsed();
    if verbose {
        eprintln!(
            "Prefill: {:.2}s ({:.1} tok/s)",
            prefill_time.as_secs_f64(),
            input_ids.len() as f64 / prefill_time.as_secs_f64()
        );
    }

    // Sample from last position
    let seq_len = logits.dim(1);
    let last_idx = Array::from_slice(&[seq_len - 1], &[1]);
    let last_logits = mlx_rs::ops::indexing::take_axis(&logits, &last_idx, 1)?;
    let last_logits = mlx_rs::ops::squeeze_axes(&last_logits, &[1])?;
    let mut next_token = sample(&last_logits, temperature, top_p, rep_penalty, &[])?;
    mlx_rs::transforms::eval(std::iter::once(&next_token))?;

    let (think_open, think_close) = tokenizer.thinking_channel_tokens();
    let mut in_thinking = false;
    let mut in_ghost_zone = false;
    // Cap consecutive ghost-zone skips so a structural marker followed by ordinary prose
    // (lowercase tokens that fail is_structural_content_start) cannot trap output forever.
    let mut ghost_skip_streak: u32 = 0;
    const MAX_GHOST_SKIPS: u32 = 4;
    let mut visible_generated: Vec<u32> = Vec::new();

    let first_tok = next_token.item::<i32>() as u32;
    let mut generated: Vec<u32> = vec![first_tok];
    if Some(first_tok) == think_open {
        in_thinking = true;
    } else if Some(first_tok) == think_close {
        in_thinking = false;
    } else if !in_thinking {
        if !in_ghost_zone || is_structural_content_start(first_tok, tokenizer) {
            visible_generated.push(first_tok);
            in_ghost_zone = false;
            if after_structural_marker(&visible_generated) { in_ghost_zone = true; }
        }
    }
    if debug_tokens {
        let tok_text = tokenizer.decode(&[first_tok]).unwrap_or_default();
        eprintln!("[tok] id={} in_thinking={} text={:?}", first_tok, in_thinking, tok_text);
    }

    let mut stdout = std::io::stdout();
    let text = tokenizer.decode(&visible_generated)?;
    if !text.is_empty() {
        match on_token.as_mut() {
            Some(f) => (*f)(&text),
            None => { print!("{}", text); stdout.flush().ok(); }
        }
    }

    // Reset perf stats and cache stats for decode-only measurement
    perf.reset();
    mem.reset_cache_stats();
    let rss_before_decode = crate::perf::current_rss_bytes();
    let t_start = Instant::now();
    let mut t_interval = Instant::now();
    let mut tokens_generated = 0usize;
    let mut prev_text_len = text.len();

    for _ in 1..max_tokens {
        let tok_id = *generated.last().unwrap();
        if tokenizer.is_eos(tok_id) {
            generated.pop();
            break;
        }

        let input = Array::from_slice(&[tok_id as i32], &[1, 1]);
        let logits = model.forward(&input, &mut cache, mem, &perf, speculate, Some(&tp))?;
        let logits = mlx_rs::ops::squeeze_axes(&logits, &[1])?;
        const REP_WINDOW: usize = 64;
        let window_start = generated.len().saturating_sub(REP_WINDOW);
        next_token = sample(&logits, temperature, top_p, rep_penalty, &generated[window_start..])?;
        mlx_rs::transforms::eval(std::iter::once(&next_token))?;

        tp.borrow_mut().end_token();

        let new_tok = next_token.item::<i32>() as u32;
        generated.push(new_tok);
        tokens_generated += 1;

        // Track thinking channel state, stream only visible tokens
        if Some(new_tok) == think_open {
            in_thinking = true;
        } else if Some(new_tok) == think_close {
            in_thinking = false;
        } else if !in_thinking {
            if !in_ghost_zone || is_structural_content_start(new_tok, tokenizer) {
                visible_generated.push(new_tok);
                in_ghost_zone = false;
                ghost_skip_streak = 0;
                // Re-enter ghost zone only when the just-pushed token completed a structural
                // marker pair. Re-checking after a filtered token would keep ghost zone on
                // forever, since visible_generated still ends with the same marker pair.
                if after_structural_marker(&visible_generated) { in_ghost_zone = true; }
            } else {
                ghost_skip_streak += 1;
                if ghost_skip_streak >= MAX_GHOST_SKIPS {
                    in_ghost_zone = false;
                    ghost_skip_streak = 0;
                }
            }
        }
        if debug_tokens {
            let tok_text = tokenizer.decode(&[new_tok]).unwrap_or_default();
            eprintln!("[tok] id={} in_thinking={} text={:?}", new_tok, in_thinking, tok_text);
        }

        let full_text = tokenizer.decode(&visible_generated)?;
        if full_text.len() > prev_text_len {
            let delta = &full_text[prev_text_len..];
            match on_token.as_mut() {
                Some(f) => (*f)(delta),
                None => { print!("{}", delta); stdout.flush().ok(); }
            }
            prev_text_len = full_text.len();
        }

        if verbose && tokens_generated % 10 == 0 {
            let elapsed = t_start.elapsed().as_secs_f64();
            let interval_elapsed = t_interval.elapsed().as_secs_f64();
            let interval_rate = 10.0 / interval_elapsed;
            eprint!(
                "\r  {} tokens, {:.1} tok/s (last 10: {:.1} tok/s)",
                tokens_generated,
                tokens_generated as f64 / elapsed,
                interval_rate,
            );
            t_interval = Instant::now();
        }
    }

    if on_token.is_none() { println!(); }
    let elapsed = t_start.elapsed().as_secs_f64();
    let rss_after_decode = crate::perf::current_rss_bytes();
    let rss_peak = crate::perf::peak_rss_bytes();
    eprintln!(
        "{} tokens in {:.1}s ({:.1} tok/s)",
        tokens_generated, elapsed,
        tokens_generated as f64 / elapsed
    );
    if verbose {
        eprintln!();

        let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        let delta = rss_after_decode as i64 - rss_before_decode as i64;
        let delta_mb = delta as f64 / (1024.0 * 1024.0);
        eprintln!(
            "RSS: {:.2} GB before decode → {:.2} GB after decode ({:+.0} MB), peak {:.2} GB",
            gb(rss_before_decode), gb(rss_after_decode), delta_mb, gb(rss_peak)
        );

        let (hits, misses, rate) = mem.take_hit_stats();
        if hits + misses > 0 {
            eprintln!(
                "Warm set hit rate: {:.1}% ({}/{} expert loads)",
                rate * 100.0, hits, hits + misses
            );
        }

        let (ch, cm, cr) = mem.take_cache_stats();
        if ch + cm > 0 {
            eprintln!(
                "Expert cache hit rate: {:.1}% ({}/{} lookups, cache size {})",
                cr * 100.0, ch, ch + cm,
                mem.cache_size(),
            );
        }

        let (touches, hits, evictions, pages_advised, madvise_errs) = mem.take_lru_stats();
        if touches > 0 {
            let rate = if touches > 0 { hits as f64 / touches as f64 } else { 0.0 };
            let mb_advised = pages_advised as f64 * 16384.0 / 1e6;
            eprintln!(
                "LRU expert cache: {:.1}% hit rate ({}/{} touches), {} evictions, {:.0} MB madvised, {} madvise errs",
                rate * 100.0, hits, touches, evictions, mb_advised, madvise_errs
            );
        }

        perf.report(tokens_generated);
        tp.borrow().report();
    }

    let recorder = tp.into_inner().recorder;
    Ok((tokenizer.decode(&visible_generated)?, recorder))
}

/// Returns true if a token is a valid content-starting token after a structural marker
/// (bullet `*   ` or section header `### `).
/// Valid starts: `**` (bold label), uppercase letter, digit, `(`, `[`.
/// Invalid (ghost): lowercase words, `-`, `#` (doubled header), whitespace-only, etc.
fn is_structural_content_start(tok: u32, tokenizer: &crate::tokenizer::QwenTokenizer) -> bool {
    if let Some(text) = tokenizer.decode_token(tok) {
        let t = text.trim_start(); // strip leading spaces
        if t.is_empty() { return false; } // whitespace-only token: skip
        let first = t.chars().next().unwrap();
        // Accept: bold `**`, any uppercase letter, digit, open paren/bracket.
        // Reject: `#` (doubled header like `### ###`), lowercase, punctuation like `-`.
        (first == '*' && t.starts_with("**"))
            || first.is_uppercase()
            || first.is_ascii_digit()
            || first == '('
            || first == '['
    } else {
        false
    }
}

/// Returns true if visible ends with a structural marker (bullet or section header)
/// that should trigger ghost-zone filtering.
fn after_structural_marker(visible: &[u32]) -> bool {
    let n = visible.len();
    if n < 2 { return false; }
    matches!(
        (visible[n-2], visible[n-1]),
        (236829, 139) | (10354, 236743) | (10354, 236829)
    )
}

fn sample(logits: &Array, temperature: f32, top_p: f32, rep_penalty: f32, recent_tokens: &[u32]) -> Result<Array, Exception> {
    // Apply repetition penalty: divide positive logits, multiply negative logits.
    // Materialises to f32 on CPU (one roundtrip) then rebuilds as bf16.
    let penalized: Option<Array> = if rep_penalty > 1.0 && !recent_tokens.is_empty() {
        let logits_f32 = logits.as_dtype(mlx_rs::Dtype::Float32)?;
        mlx_rs::transforms::eval(std::iter::once(&logits_f32))?;
        let src: &[f32] = logits_f32.as_slice();
        let mut data = src.to_vec();
        let shape: Vec<i32> = logits_f32.shape().iter().map(|&x| x as i32).collect();
        let vocab_size = *shape.last().unwrap() as usize;
        let offset = data.len() - vocab_size;
        let mut seen = std::collections::HashSet::new();
        for &tok in recent_tokens {
            if seen.insert(tok) {
                let idx = offset + tok as usize;
                if idx < data.len() {
                    if data[idx] >= 0.0 { data[idx] /= rep_penalty; } else { data[idx] *= rep_penalty; }
                }
            }
        }
        let arr = Array::from_slice(&data, &shape);
        Some(arr.as_dtype(logits.dtype())?)
    } else {
        None
    };
    let logits: &Array = penalized.as_ref().unwrap_or(logits);

    if temperature == 0.0 {
        return mlx_rs::ops::indexing::argmax_axis(logits, -1, None);
    }

    let logits = logits / temperature;

    let logits = if top_p < 1.0 {
        let neg_logits = -&logits;
        let sorted_indices = mlx_rs::ops::argsort_axis(&neg_logits, -1)?;
        let sorted_logits = mlx_rs::ops::indexing::take_along_axis(&logits, &sorted_indices, Some(-1))?;
        let probs = mlx_rs::ops::softmax_axis(&sorted_logits, -1, None)?;
        let cumulative = mlx_rs::ops::cumsum(&probs, Some(-1), None, None)?;
        let diff = &cumulative - &probs;
        let threshold = Array::from_f32(top_p);
        let mask = diff.gt(&threshold)?;
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        let filtered = mlx_rs::ops::r#where(&mask, &neg_inf, &sorted_logits)?;
        let inv_indices = mlx_rs::ops::argsort_axis(&sorted_indices, -1)?;
        mlx_rs::ops::indexing::take_along_axis(&filtered, &inv_indices, Some(-1))?
    } else {
        logits
    };

    let probs = mlx_rs::ops::softmax_axis(&logits, -1, None)?;
    let log_probs = mlx_rs::ops::log(&(&probs + 1e-10f32))?;
    mlx_rs::random::categorical(&log_probs, Some(-1), None::<mlx_rs::random::ShapeOrCount>, None::<&Array>)
}
