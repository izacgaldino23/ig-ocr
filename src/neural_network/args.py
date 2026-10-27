import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Neural network training")
    parser.add_argument("-i", "--input", type=float, default=0, help="Input model number")
    parser.add_argument("-o", "--output", type=float, default=0, help="Output model number")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (defaults to 1024 on CUDA and 64 on CPU)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Data-loader worker count (defaults to 4 on CUDA and 1 on CPU)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for repeatable comparisons"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Train the reduced-channel CNN variant",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision (FP16) with gradient scaling",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile five batches in the second epoch",
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers cannot be negative")

    return args
