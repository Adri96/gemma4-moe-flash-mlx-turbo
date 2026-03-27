#!/usr/bin/env python3
"""CLI for flash-loaded Qwen3.5-35B-A3B inference."""

import argparse
import sys


def cmd_split(args):
    from flash_qwen.split import main as split_main
    sys.argv = ["split", "--model-path", args.model_path, "--output-path", args.output_path]
    split_main()


def cmd_generate(args):
    from flash_qwen.engine import FlashInferenceEngine

    engine = FlashInferenceEngine(
        split_model_path=args.model_path,
        cache_size_mb=args.cache_size_mb,
        original_model_path=args.tokenizer_path,
        kv_bits=args.kv_bits,
    )

    if args.interactive:
        print("\nInteractive mode (type 'quit' to exit)")
        print("-" * 50)
        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            # Apply chat template if available
            if hasattr(engine.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted = engine.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt

            output = engine.generate(
                formatted,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"\nAssistant: {output}")
    else:
        prompt = args.prompt
        if not prompt:
            print("Error: --prompt required in non-interactive mode", file=sys.stderr)
            sys.exit(1)

        if hasattr(engine.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        output = engine.generate(
            formatted,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(output)


def main():
    parser = argparse.ArgumentParser(
        description="Flash-loaded Qwen3.5-35B-A3B inference"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Split command
    split_parser = subparsers.add_parser("split", help="Split model into resident + expert files")
    split_parser.add_argument("--model-path", required=True, help="Original MLX model path")
    split_parser.add_argument("--output-path", required=True, help="Output directory")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--model-path", required=True, help="Path to split model")
    gen_parser.add_argument("--tokenizer-path", help="Path to tokenizer (if different from model)")
    gen_parser.add_argument("--prompt", help="Input prompt")
    gen_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive chat mode")
    gen_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    gen_parser.add_argument("--cache-size-mb", type=int, default=6144, help="Expert LRU cache size in MB")
    gen_parser.add_argument("--kv-bits", type=int, choices=[2, 3, 4], default=None,
                            help="TurboQuant KV cache compression bits (default: None = FP16)")

    args = parser.parse_args()

    if args.command == "split":
        cmd_split(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
