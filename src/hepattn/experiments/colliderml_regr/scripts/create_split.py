#!/usr/bin/env python3
"""Create a deterministic train/val/test split file for preprocessed shards.

Discovers all ``shard_XXXX/`` directories in the preprocessed directory and
assigns each to train, val, or test based on the given fractions.  The result
is saved as ``split.json`` inside the preprocessed directory so that all
subsequent training runs use a consistent split regardless of how many shards
they actually load.

Usage::

    # Default 90/5/5 split
    python create_split.py --preprocessed-dir ${DATA_ROOT}/p0_preprocessed

    # Custom fractions
    python create_split.py --preprocessed-dir ${DATA_ROOT}/p0_preprocessed \
        --train-frac 0.8 --val-frac 0.1 --test-frac 0.1

    # Dry run (print only, don't write)
    python create_split.py --preprocessed-dir ${DATA_ROOT}/p0_preprocessed --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def create_split(
    preprocessed_dir: str | Path,
    train_frac: float = 0.9,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, list[int]]:
    """Create and save a shard split file.

    Shards are sorted by index and split deterministically (no shuffle)
    so the assignment is reproducible without needing the seed at load time.

    Parameters
    ----------
    preprocessed_dir : str | Path
        Root directory containing ``shard_XXXX/`` subdirectories.
    train_frac, val_frac, test_frac : float
        Fractions for each split.  Must sum to 1.0 (within tolerance).
    seed : int
        Stored in the file for provenance but not used for splitting
        (split is deterministic by sorted shard order).
    dry_run : bool
        If True, print the split but don't write the file.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to shard index lists.
    """
    preprocessed_dir = Path(preprocessed_dir)
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total:.6f}")

    # Discover shards
    shard_dirs = sorted(preprocessed_dir.glob("shard_*"))
    shard_indices = [int(d.name.split("_")[1]) for d in shard_dirs]
    n = len(shard_indices)

    if n < 3:
        raise ValueError(f"Need at least 3 shards to create a split, found {n}")

    # Deterministic contiguous split
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val

    split = {
        "train": shard_indices[:n_train],
        "val": shard_indices[n_train : n_train + n_val],
        "test": shard_indices[n_train + n_val :],
    }

    print(f"Preprocessed dir : {preprocessed_dir}")
    print(f"Total shards     : {n}")
    print(f"Train shards     : {len(split['train'])}  (indices {split['train'][0]}–{split['train'][-1]})")
    print(f"Val shards       : {len(split['val'])}  (indices {split['val'][0]}–{split['val'][-1]})")
    print(f"Test shards      : {len(split['test'])}  (indices {split['test'][0]}–{split['test'][-1]})")

    payload = {
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "_meta": {
            "n_total": n,
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
            "seed": seed,
        },
    }

    out_path = preprocessed_dir / "split.json"
    if dry_run:
        print(f"\n[DRY RUN] Would write to: {out_path}")
        print(json.dumps(payload, indent=2))
    else:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSplit saved to: {out_path}")

    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test shard split file")
    parser.add_argument("--preprocessed-dir", type=str, required=True,
                        help="Path to preprocessed shard directory")
    parser.add_argument("--train-frac", type=float, default=0.9)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print split without writing file")
    args = parser.parse_args()

    create_split(
        preprocessed_dir=args.preprocessed_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
