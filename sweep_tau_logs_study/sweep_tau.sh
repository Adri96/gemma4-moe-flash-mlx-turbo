#!/usr/bin/env bash
# τ sweep: sparse-V threshold × context size
# Candidates: τ ∈ {0.0(baseline), 5e-5, 1e-4, 3e-4, 1e-3}
# Contexts:   ~2K, ~4K, ~8K prompt tokens

set -euo pipefail

BINARY="./target/release/flash-moe"
MODEL_PATH="./split_gemma4_ud4"
TOKENIZER_PATH="./gemma4-ud-4bit"
DECODE_TOKENS=80
RESULTS_FILE="sweep_tau_results.txt"
LOG_DIR="sweep_tau_logs"
mkdir -p "$LOG_DIR"

THRESHOLDS=(0.0 5e-5 1e-4 3e-4 1e-3)

# ~210 words ≈ ~315 tokens per paragraph (1.5 tok/word).
# Repetitions: 2K→7, 4K→13, 8K→26
SEED="The universe is a vast and mysterious place filled with countless stars, planets, and galaxies that stretch far beyond human imagination. Scientists have devoted entire careers to studying the cosmos in an effort to understand its origins, structure, and ultimate fate. From the revolutionary Big Bang theory to the enigmatic concepts of dark matter and dark energy, modern cosmology has made remarkable strides in explaining the fundamental nature of reality. Yet many profound questions remain unanswered, and new discoveries continue to reshape our understanding in unexpected ways. The search for extraterrestrial life, the nature of black holes, and the tantalizing possibility of parallel universes are just a few of the mysteries driving scientific inquiry ever forward. As telescopes grow more powerful and computational methods more sophisticated, humanity stands on the brink of potentially revolutionary discoveries that could forever transform our view of our place in the cosmos. Each new observation brings fresh insight and raises new questions, reminding us that nature is far more complex and wondrous than we had imagined. The interplay between theory and observation has always been central to scientific progress, and nowhere is this more apparent than in cosmology itself. Galaxies collide and merge over billions of years, stars are born in vast molecular clouds and die in spectacular supernovae, and neutron stars spin hundreds of times per second sending pulses of radiation across the universe. The cosmic web of filaments and voids that spans the observable universe emerged from quantum fluctuations in the earliest moments after the Big Bang, a testament to the deep connections between the very large and the very small scales of physical reality."

repeat_text() {
    local n=$1
    local out=""
    for ((i=0; i<n; i++)); do out="$out $SEED"; done
    echo "$out"
}

echo "τ sweep started: $(date)"
echo ""
printf "%-8s  %-8s  %8s  %s\n" "Context" "tau" "tok/s" "prefill_toks" > "$RESULTS_FILE"
printf "%-8s  %-8s  %8s  %s\n" "-------" "--------" "--------" "------------" >> "$RESULTS_FILE"

for ctx_label in "2K" "4K" "8K"; do
    case "$ctx_label" in
        2K) reps=7  ;;
        4K) reps=13 ;;
        8K) reps=26 ;;
    esac
    prompt=$(repeat_text $reps)

    for tau in "${THRESHOLDS[@]}"; do
        tag="${ctx_label}_tau${tau}"
        log="$LOG_DIR/${tag}.log"

        echo "=== ctx=$ctx_label  τ=$tau ===" | tee "$log"

        set +e
        "$BINARY" generate \
            --model-path  "$MODEL_PATH" \
            --tokenizer-path "$TOKENIZER_PATH" \
            --prompt "$prompt" \
            --max-tokens "$DECODE_TOKENS" \
            --temperature 0.7 \
            --sparse-v-threshold "$tau" \
            --kv-quant-bits 3 \
            --stats \
            > /dev/null 2>> "$log"
        EXIT=$?
        set -e

        if [[ $EXIT -ne 0 ]]; then
            echo "  FAILED (exit $EXIT)" | tee -a "$log"
            printf "%-8s  %-8s  %8s  %s\n" "$ctx_label" "$tau" "FAILED" "-" >> "$RESULTS_FILE"
            continue
        fi

        # "N tokens in Xs (Y tok/s)" — always printed unconditionally by engine.rs
        tps=$(grep -oE '[0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s\)' "$log" \
              | tail -1 \
              | grep -oE '\([0-9.]+ tok/s\)' \
              | grep -oE '[0-9.]+' || echo "?")

        # "Prefilling N tokens..." — printed when --stats
        prefill=$(grep -oE 'Prefilling [0-9]+ tokens' "$log" \
                  | grep -oE '[0-9]+' || echo "?")

        echo "  → prefill=${prefill} toks  decode=${tps} tok/s"
        printf "%-8s  %-8s  %8s  %s\n" "$ctx_label" "$tau" "$tps" "$prefill" >> "$RESULTS_FILE"

        # Brief cool-down between runs (let Metal/GPU settle)
        sleep 5
    done
done

echo ""
echo "===== SWEEP RESULTS ====="
cat "$RESULTS_FILE"
echo ""
echo "τ sweep finished: $(date)"
echo "Full logs in: $LOG_DIR/"
