"""Stage 4: MinHash LSH near-deduplication per language."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH

from terminus.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def _ngrams(text: str, n: int) -> list:
    """Generate character n-grams from text."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _make_minhash(text: str, num_perm: int, ngram_size: int) -> MinHash:
    """Create a MinHash signature from text n-grams."""
    m = MinHash(num_perm=num_perm)
    for gram in _ngrams(text, ngram_size):
        m.update(gram.encode("utf-8"))
    return m


def run(input_path: Path, output_dir: Path, config: dict) -> Path:
    """Deduplicate documents using MinHash LSH, per language.

    Args:
        input_path: Path to heuristics JSONL.
        output_dir: Directory to write output JSONL.
        config: Dedup config dict (num_perm, threshold, ngram_size).

    Returns:
        Path to the output JSONL file.
    """
    output_path = output_dir / "04_dedup.jsonl"
    num_perm = config.get("num_perm", 128)
    threshold = config.get("threshold", 0.7)
    ngram_size = config.get("ngram_size", 5)

    # Group documents by language
    by_lang = defaultdict(list)
    for doc in read_jsonl(input_path):
        lang = doc.get("lang_ft", doc.get("lang_cc", "unk"))
        by_lang[lang].append(doc)

    kept = []
    for lang in sorted(by_lang):
        docs = by_lang[lang]
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        lang_kept = []
        duplicates = 0

        for doc in docs:
            doc_id = doc["id"]
            mh = _make_minhash(doc["text"], num_perm, ngram_size)

            if lsh.query(mh):
                duplicates += 1
                continue

            try:
                lsh.insert(doc_id, mh)
            except ValueError:
                # Duplicate key — skip
                duplicates += 1
                continue

            lang_kept.append(doc)

        kept.extend(lang_kept)
        logger.info(
            "  %s: %d → %d (removed %d duplicates, %.1f%%)",
            lang, len(docs), len(lang_kept), duplicates,
            100.0 * duplicates / max(len(docs), 1),
        )

    n = write_jsonl(output_path, kept)
    logger.info("Dedup complete — %d documents retained", n)
    return output_path
