from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "datasets": "datasets",
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "seaborn": "seaborn",
    "torch": "torch",
    "tqdm": "tqdm",
    "transformers": "transformers",
}


def find_missing_modules() -> list[str]:
    missing = []
    for module_name, package_name in REQUIRED_MODULES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-platform launcher for the SML NLP project.",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=["part1", "part2", "part3", "compare-encoders"],
        help="Run Part 1, Part 2, Part 3, or compare all cached encoders.",
    )
    parser.add_argument(
        "--encoder",
        default="microsoft/deberta-v3-base",
        help="Encoder-only model to use for Part 2 or Part 3. Must output 768-dim hidden states.",
    )
    return parser


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    script_path = root_dir / "code" / "cli.py"
    requirements_path = root_dir / "requirements.txt"
    args = build_parser().parse_args()

    env = os.environ.copy()
    pythonpath_parts = [str(root_dir / "code")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    mpl_dir = root_dir / "results" / "cache" / "mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    print("Running SML NLP project...")
    print(f"Project root: {root_dir}")

    missing = find_missing_modules()
    if missing:
        print()
        print("Missing Python packages detected.")
        print(f"Install them with: {sys.executable} -m pip install -r {requirements_path}")
        print("After that, run one of these:")
        print(f"{sys.executable} {Path(__file__).name} part1")
        print(f"{sys.executable} {Path(__file__).name} part2 --encoder microsoft/deberta-v3-base")
        print(f"{sys.executable} {Path(__file__).name} part3 --encoder microsoft/deberta-v3-base")
        print(f"{sys.executable} {Path(__file__).name} compare-encoders")
        return 1

    if args.stage is None:
        print()
        print("Use one stage at a time:")
        print(f"{sys.executable} {Path(__file__).name} part1")
        print(f"{sys.executable} {Path(__file__).name} part2 --encoder microsoft/deberta-v3-base")
        print(f"{sys.executable} {Path(__file__).name} part3 --encoder microsoft/deberta-v3-base")
        print(f"{sys.executable} {Path(__file__).name} compare-encoders")
        print()
        print("Recommended workflow:")
        print("1. Run part1 for sparse features and classical ML.")
        print("2. Run part2 separately on Kaggle or Colab GPU to create cached 768-dim embeddings.")
        print("3. Run compare-encoders after caching multiple encoders to choose the strongest one.")
        print("4. Run part3 with the best encoder to train classical ML models on the cached embeddings.")
        return 0

    print("If this is the first embedding run, the dataset and encoder model may download once.")

    command = [sys.executable, str(script_path), args.stage]
    if args.stage in {"part2", "part3"}:
        command.extend(["--encoder", args.encoder])
    result = subprocess.run(command, env=env)
    if result.returncode == 0:
        print()
        print("Done.")
        print(f"Report: {root_dir / 'report.pdf'}")
        print(f"Figures: {root_dir / 'results' / 'figures'}")
        print(f"Tables: {root_dir / 'results' / 'tables'}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
