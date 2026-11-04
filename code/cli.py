from __future__ import annotations

import argparse

from encoder_registry import DEFAULT_ENCODER, get_supported_encoders


def run_part1() -> None:
    from part1_sparse import run_sparse_part

    run_sparse_part()


def run_part2(encoder: str) -> None:
    from part2_embeddings import run_embedding_part

    run_embedding_part(model_name=encoder)


def run_part3(encoder: str) -> None:
    from part3_contextual import run_contextual_part

    run_contextual_part(model_name=encoder)


def run_compare_encoders() -> None:
    from encoder_comparison import compare_supported_encoders

    compare_supported_encoders()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-based runner for the SML NLP project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    part1 = subparsers.add_parser("part1", help="Run sparse feature pipeline only.")
    part1.set_defaults(func=lambda args: run_part1())

    part2 = subparsers.add_parser("part2", help="Extract and cache encoder embeddings only.")
    part2.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        choices=get_supported_encoders(),
        help="Encoder-only model to use for 768-dim embeddings.",
    )
    part2.set_defaults(func=lambda args: run_part2(args.encoder))

    part3 = subparsers.add_parser("part3", help="Run classical ML on cached contextual embeddings.")
    part3.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        choices=get_supported_encoders(),
        help="Encoder whose cached embeddings should be used.",
    )
    part3.set_defaults(func=lambda args: run_part3(args.encoder))

    compare_encoders = subparsers.add_parser(
        "compare-encoders",
        help="Evaluate all cached encoder embeddings and rank them.",
    )
    compare_encoders.set_defaults(func=lambda args: run_compare_encoders())
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
