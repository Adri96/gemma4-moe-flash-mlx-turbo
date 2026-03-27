"""CLI to split a model into resident weights + per-layer expert files."""

import argparse
import sys

from flash_qwen._native import split_model


def main():
    parser = argparse.ArgumentParser(
        description="Split Qwen3.5 MoE model into resident weights + expert files for flash loading"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the original MLX model directory",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output directory for the split model",
    )
    args = parser.parse_args()

    print(f"Splitting model: {args.model_path}")
    print(f"Output: {args.output_path}")
    split_model(args.model_path, args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
