"""Stage 5: Stratified quadrant sampling for the LLM judge."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path

from terminus.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

QUADRANT_LABELS = {
    ("en", True): "A (en, pass)",
    ("en", False): "B (en, fail)",
    ("ru", True): "C (ru, pass)",
    ("ru", False): "D (ru, fail)",
}


def run(input_path: Path, output_dir: Path, config: dict) -> Path:
    """Stratified quadrant sampling for the judge stage.

    Quadrants:
        A: heuristic_pass=True,  lang=en
        B: heuristic_pass=False, lang=en
        C: heuristic_pass=True,  lang=ru
        D: heuristic_pass=False, lang=ru

    Args:
        input_path: Path to dedup JSONL.
        output_dir: Directory to write output JSONL.
        config: Sampling config dict (judge_sample_size, per_quadrant).

    Returns:
        Path to the output JSONL file.
    """
    output_path = output_dir / "05_sampled.jsonl"
    per_quadrant = config.get("per_quadrant", 75)

    # Bucket into quadrants
    quadrants = defaultdict(list)
    for doc in read_jsonl(input_path):
        lang = doc.get("lang_ft", doc.get("lang_cc", "unk"))
        passed = doc.get("heuristic_pass", False)
        quadrants[(lang, passed)].append(doc)

    # Log sizes before sampling
    logger.info("Quadrant sizes before sampling:")
    for key in [("en", True), ("en", False), ("ru", True), ("ru", False)]:
        label = QUADRANT_LABELS.get(key, str(key))
        logger.info("  %s: %d docs", label, len(quadrants[key]))

    # Sample from each quadrant
    sampled = []
    random.seed(42)

    logger.info("Sampling up to %d per quadrant:", per_quadrant)
    for key in [("en", True), ("en", False), ("ru", True), ("ru", False)]:
        pool = quadrants[key]
        n = min(per_quadrant, len(pool))
        if len(pool) > per_quadrant:
            picked = random.sample(pool, per_quadrant)
        else:
            picked = pool

        label = QUADRANT_LABELS.get(key, str(key))
        logger.info("  %s: %d → %d", label, len(pool), n)

        # Tag each doc with its quadrant
        for doc in picked:
            doc["quadrant"] = label
        sampled.extend(picked)

    n = write_jsonl(output_path, sampled)
    logger.info("Sampled %d documents total", n)
    return output_path
