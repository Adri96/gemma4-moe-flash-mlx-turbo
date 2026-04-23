mod api_types;
mod cache;
mod config;
mod engine;
mod ffi;
mod memory;
mod model;
mod perf;
mod server;
mod splitter;
mod tokenizer;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "flash-moe")]
#[command(about = "All-Rust UMA-native sparse MoE inference on Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Split original model into resident weights + per-layer expert files
    Split {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        output_path: PathBuf,
        /// Expert file format (only "ecb" is supported)
        #[arg(long, default_value = "ecb")]
        format: String,
    },
    /// Start an OpenAI-compatible HTTP API server
    Serve {
        #[arg(long, default_value = "./split_gemma4_ud4")]
        model_path: PathBuf,
        #[arg(long, default_value = "./split_gemma4_ud4")]
        tokenizer_path: PathBuf,
        /// Host to bind (use 0.0.0.0 to allow LAN access)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Port to listen on (11434 = Ollama default, 1234 = LM Studio default)
        #[arg(long, default_value_t = 11434)]
        port: u16,
        /// Model ID reported to clients via /v1/models
        #[arg(long, default_value = "flash-moe")]
        model_id: String,
        #[arg(long, default_value = "3")]
        kv_quant_bits: Option<u8>,
        #[arg(long)]
        no_kv_quant: bool,
        #[arg(long, default_value_t = 1.15)]
        rep_penalty: f32,
        #[arg(long)]
        expert_cache: Option<usize>,
        #[arg(long)]
        warm_set: bool,
    },
    /// Generate text from a prompt
    Generate {
        #[arg(long, default_value = "./split_gemma4")]
        model_path: PathBuf,
        #[arg(long, default_value = "./split_gemma4")]
        tokenizer_path: PathBuf,
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value_t = 2048)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        #[arg(long)]
        warm_experts: Option<PathBuf>,
        /// TurboQuant KV cache quantization bits (2, 3, or 4). Default: 3.
        #[arg(long, default_value = "3")]
        kv_quant_bits: Option<u8>,
        /// Disable TurboQuant KV cache quantization (use plain bf16).
        #[arg(long)]
        no_kv_quant: bool,
        /// Disable speculative prefetch for next-layer predicted experts.
        #[arg(long)]
        no_speculate: bool,
        /// Load warm set at startup (preads frequent experts into page cache).
        #[arg(long)]
        warm_set: bool,
        /// Calibrate co-occurrence predictor using N tokens, save to model dir.
        #[arg(long)]
        calibrate: Option<usize>,
        /// Print performance statistics after generation.
        #[arg(long)]
        stats: bool,
        /// Interactive multi-turn chat mode (reads turns from stdin).
        #[arg(long)]
        chat: bool,
        /// Print each generated token ID and text (for debugging output artifacts).
        #[arg(long)]
        debug_tokens: bool,
        /// Repetition penalty applied to recently generated tokens (1.0 = disabled).
        #[arg(long, default_value_t = 1.15)]
        rep_penalty: f32,
        /// LRU expert page cache size (number of experts to keep warm in RAM).
        /// Each expert is ~3.35 MB. Without this flag all accessed experts accumulate
        /// in the OS page cache (up to 12-13 GB). Example: --expert-cache 500 (~1.68 GB).
        #[arg(long)]
        expert_cache: Option<usize>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Split {
            model_path,
            output_path,
            format,
        } => {
            eprintln!("Splitting model: {} → {} (format: {})", model_path.display(), output_path.display(), format);
            splitter::split_model(&model_path, &output_path, &format)?;
            eprintln!("Done.");
        }

        Command::Serve {
            model_path,
            tokenizer_path,
            host,
            port,
            model_id,
            kv_quant_bits,
            no_kv_quant,
            rep_penalty,
            expert_cache,
            warm_set,
        } => {
            let kv_quant_bits = if no_kv_quant { None } else { kv_quant_bits };

            let config_path = model_path.join("config.json");
            let (args, quant) = config::TextModelArgs::from_config_file(&config_path)?;

            eprintln!("Loading tokenizer from {}...", tokenizer_path.display());
            let tokenizer = tokenizer::QwenTokenizer::from_dir(&tokenizer_path)?;

            let expert_dir = model_path.join("experts");
            eprintln!("Mapping expert files from {}...", expert_dir.display());
            let mut mem_mgr = memory::ExpertMemoryManager::new(&expert_dir, args.num_hidden_layers)?;
            if let Some(cap) = expert_cache {
                mem_mgr.set_expert_cache(cap);
                eprintln!("Expert LRU cache: {} experts (~{:.0} MB)", cap, cap as f64 * 3.35);
            }

            eprintln!("Loading model from {}...", model_path.display());
            let model = model::load_model(&model_path, &args, quant.as_ref(), true)?;

            // Warm set prefetch (opt-in)
            if warm_set {
                let auto = model_path.join("warm_experts.json");
                if auto.exists() {
                    eprintln!("Prefetching warm set from {}...", auto.display());
                    let warm: serde_json::Value =
                        serde_json::from_str(&std::fs::read_to_string(&auto)?)?;
                    let experts: Vec<(u32, u32)> = warm["experts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|pair| {
                            let arr = pair.as_array().unwrap();
                            (arr[0].as_u64().unwrap() as u32, arr[1].as_u64().unwrap() as u32)
                        })
                        .collect();
                    mem_mgr.mlock_warm_set(&experts);
                    mem_mgr.set_warm_set(&experts);
                }
            }

            if let Some(bits) = kv_quant_bits {
                eprintln!("TurboQuant KV cache: {}-bit", bits);
            }
            eprintln!("Model ready. Starting server...");

            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(server::run(
                &host, port, model, tokenizer, mem_mgr, model_id, kv_quant_bits, rep_penalty,
            ))?;
        }

        Command::Generate {
            model_path,
            tokenizer_path,
            prompt,
            max_tokens,
            temperature,
            top_p,
            warm_experts,
            kv_quant_bits,
            no_kv_quant,
            no_speculate,
            warm_set,
            calibrate,
            stats,
            chat,
            debug_tokens,
            expert_cache,
            rep_penalty,
        } => {
            let kv_quant_bits = if no_kv_quant { None } else { kv_quant_bits };
            // Load config
            let config_path = model_path.join("config.json");
            let (args, quant) = config::TextModelArgs::from_config_file(&config_path)?;

            // Load tokenizer
            if stats { eprintln!("Loading tokenizer from {}...", tokenizer_path.display()); }
            let tokenizer = tokenizer::QwenTokenizer::from_dir(&tokenizer_path)?;

            // Apply chat template
            let make_prompt = |messages: &[tokenizer::ChatMessage]| -> String {
                tokenizer.apply_chat_template(messages).unwrap_or_else(|_| {
                    let body: String = messages.iter().map(|m| {
                        format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content)
                    }).collect();
                    format!("<bos>{}<|im_start|>assistant\n", body)
                })
            };

            let mut messages = vec![tokenizer::ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            }];
            let chat_prompt = make_prompt(&messages);

            if debug_tokens {
                // Show the last 120 chars of the rendered prompt to verify template output
                let tail_start = chat_prompt.len().saturating_sub(120);
                eprintln!("[debug] prompt tail: {:?}", &chat_prompt[tail_start..]);
                let (to, tc) = tokenizer.thinking_channel_tokens();
                eprintln!("[debug] thinking_open={:?}  thinking_close={:?}", to, tc);
            }

            // Create ExpertMemoryManager — mmaps expert files, used for on-demand loading
            let expert_dir = model_path.join("experts");
            if stats { eprintln!("Mapping expert files from {}...", expert_dir.display()); }
            let mut mem_mgr = memory::ExpertMemoryManager::new(&expert_dir, args.num_hidden_layers)?;
            if let Some(cap) = expert_cache {
                mem_mgr.set_expert_cache(cap);
                if stats {
                    eprintln!(
                        "Expert LRU cache: {} experts (~{:.0} MB)",
                        cap,
                        cap as f64 * 3.35
                    );
                }
            }

            // Load model FIRST (resident weights only — no expert arrays).
            // This allocates 2.76 GB of Metal buffers. Loading before warm set
            // prefetch ensures madvise pages aren't evicted by weight allocation.
            if stats { eprintln!("Loading model from {}...", model_path.display()); }
            let mut model = model::load_model(&model_path, &args, quant.as_ref(), stats)?;

            // Prefetch warm set into page cache AFTER model loading
            let warm_path = warm_experts
                .or_else(|| {
                    let auto = model_path.join("warm_experts.json");
                    if auto.exists() { Some(auto) } else { None }
                });

            if let Some(wp) = warm_path.filter(|_| warm_set) {
                if stats { eprintln!("Prefetching warm set from {}...", wp.display()); }
                let warm: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&wp)?)?;
                let experts: Vec<(u32, u32)> = warm["experts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| {
                        let arr = pair.as_array().unwrap();
                        (arr[0].as_u64().unwrap() as u32, arr[1].as_u64().unwrap() as u32)
                    })
                    .collect();

                let advised = mem_mgr.mlock_warm_set(&experts);
                mem_mgr.set_warm_set(&experts);
                if stats {
                    eprintln!(
                        "Warm set: {} experts, prefetched {:.1} GB",
                        experts.len(),
                        advised as f64 / 1e9
                    );
                }
            }

            // Load co-occurrence predictor if available
            let cooccur_path = model_path.join("cooccurrence.bin");
            let cooccur = if calibrate.is_none() && cooccur_path.exists() {
                match model::moe::CooccurrencePredictor::load(&cooccur_path) {
                    Ok(p) => {
                        if stats { eprintln!("Loaded co-occurrence predictor from {}", cooccur_path.display()); }
                        Some(p)
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to load co-occurrence predictor: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            // Set up calibration recorder if requested
            let recorder = calibrate.map(|_| {
                let num_experts = args.num_experts;
                if stats { eprintln!("Calibration mode: recording routing decisions"); }
                model::moe::CalibrationRecorder::new(args.num_hidden_layers, num_experts)
            });

            // Speculation is OFF by default for Gemma 4. Level C prediction's virtual KV
            // concat in forward_speculative grows linearly with cache length and burns
            // ~24 ms of GPU compute per token for near-zero accuracy benefit (commit
            // 10ecf97). At long contexts (200+ tokens) this manifests as a generation
            // hang. `--no-speculate` is kept as a no-op for backwards compat.
            let _ = no_speculate;
            let speculate = false;
            if stats {
                if let Some(bits) = kv_quant_bits {
                    eprintln!("TurboQuant KV cache: {}-bit", bits);
                }
                eprintln!("Engine ready.");
                if !speculate {
                    eprintln!("Speculative prediction: disabled");
                }
            }
            let max_tokens = calibrate.unwrap_or(max_tokens);
            let (output, recorder) = engine::generate(
                &mut model,
                &tokenizer,
                &chat_prompt,
                max_tokens,
                temperature,
                top_p,
                rep_penalty,
                &mem_mgr,
                kv_quant_bits,
                speculate,
                cooccur,
                recorder,
                stats,
                debug_tokens,
                None,
            )?;

            // Save calibration results
            if let Some(recorder) = recorder {
                let predictor = recorder.build_predictor(12);
                predictor.save(&cooccur_path)?;
                if stats { eprintln!("Saved co-occurrence predictor to {}", cooccur_path.display()); }
            }

            // Multi-turn chat loop
            if chat {
                use std::io::{BufRead, Write};
                messages.push(tokenizer::ChatMessage {
                    role: "assistant".to_string(),
                    content: output,
                });
                let stdin = std::io::stdin();
                loop {
                    print!("\n> ");
                    std::io::stdout().flush().ok();
                    let mut line = String::new();
                    if stdin.lock().read_line(&mut line)? == 0 { break; }
                    let line = line.trim().to_string();
                    if line.is_empty() || line == "/quit" { break; }
                    messages.push(tokenizer::ChatMessage {
                        role: "user".to_string(),
                        content: line,
                    });
                    let turn_prompt = make_prompt(&messages);
                    let (turn_output, _) = engine::generate(
                        &mut model,
                        &tokenizer,
                        &turn_prompt,
                        max_tokens,
                        temperature,
                        top_p,
                        rep_penalty,
                        &mem_mgr,
                        kv_quant_bits,
                        speculate,
                        None,
                        None,
                        stats,
                        debug_tokens,
                        None,
                    )?;
                    messages.push(tokenizer::ChatMessage {
                        role: "assistant".to_string(),
                        content: turn_output,
                    });
                }
            }
        }
    }

    Ok(())
}
