#!/usr/bin/env python3
"""CLI: index a folder of documents.

    python -m scripts.ingest --docs data/docs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.ingest import ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index documents for the assistant.")
    parser.add_argument("--docs", default="data/docs", help="folder of documents")
    parser.add_argument(
        "--skip-index-setup",
        action="store_true",
        help="assume the search index already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not get_settings().configured:
        print("Azure credentials are missing. Copy .env.example to .env first.")
        return 1

    docs_dir = Path(args.docs)
    if not docs_dir.is_dir():
        print(f"No such directory: {docs_dir}")
        return 1

    stats = ingest(docs_dir, recreate_index=not args.skip_index_setup)
    print(
        f"Indexed {stats['chunks']} chunks from {stats['documents']} documents "
        f"({stats['uploaded']} uploaded)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
