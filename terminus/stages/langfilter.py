"""Stage 2: FastText language detection and balanced sampling."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List

import fasttext

from terminus.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

# Suppress FastText's load warning about deprecated API
fasttext.FastText.eprint = lambda x: None

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_NAME = "lid.176.bin"


def _get_model_path() -> Path:
    """Return path to the FastText LID model, downloading if needed."""
    # Check common locations
    candidates = [
        Path.cwd() / MODEL_NAME,
        Path(__file__).resolve().parent.parent.parent / MODEL_NAME,
        Path.home() / ".cache" / "fasttext" / MODEL_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Download to cache
    cache_dir = Path.home() / ".cache" / "fasttext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / MODEL_NAME

    logger.info("Downloading FastText LID model to %s ...", dest)
    import requests
    from tqdm import tqdm

    resp = requests.get(MODEL_URL, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="lid.176.bin") as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def _detect_lang(model, text: str) -> tuple:
    """Detect language using FastText. Returns (lang_code, confidence)."""
    # FastText expects a single line
    clean = text.replace("\n", " ")[:5000]
    # Use the internal C++ predict to avoid numpy 2.x compat issue
    # Returns list of (prob, label) tuples
    results = model.f.predict(clean, 1, 0.0, "")
    if results:
        confidence, raw_label = results[0]
        label = raw_label.replace("__label__", "")
    else:
        label = "unk"
        confidence = 0.0
    return label, confidence


def run(input_path: Path, output_dir: Path, languages: List[str], sample_size: int) -> Path:
    """Filter documents by language and sample to equal sizes.

    Args:
        input_path: Path to ingest JSONL.
        output_dir: Directory to write output JSONL.
        languages: List of ISO 639-1 language codes to keep.
        sample_size: Target number of documents per language.

    Returns:
        Path to the output JSONL file.
    """
    output_path = output_dir / "02_langfilter.jsonl"

    model_path = _get_model_path()
    logger.info("Loading FastText model from %s", model_path)
    model = fasttext.load_model(str(model_path))

    # Bucket documents by detected language
    buckets = defaultdict(list)
    confidences = defaultdict(list)
    total = 0
    lang_set = set(languages)

    for doc in read_jsonl(input_path):
        total += 1
        lang, conf = _detect_lang(model, doc["text"])

        if lang not in lang_set:
            continue

        doc["lang_ft"] = lang
        doc["lang_ft_conf"] = round(conf, 4)
        buckets[lang].append(doc)
        confidences[lang].append(conf)

    # Log detection stats
    logger.info("Total documents scanned: %d", total)
    for lang in languages:
        docs = buckets.get(lang, [])
        confs = confidences.get(lang, [])
        if confs:
            mean_conf = sum(confs) / len(confs)
            min_conf = min(confs)
            p25 = sorted(confs)[len(confs) // 4]
            logger.info(
                "  %s: %d docs | confidence mean=%.3f min=%.3f p25=%.3f",
                lang, len(docs), mean_conf, min_conf, p25,
            )
        else:
            logger.info("  %s: 0 docs", lang)

    # Sample to equal sizes
    min_available = min((len(buckets[l]) for l in languages if buckets[l]), default=0)
    target = min(sample_size, min_available)
    logger.info("Sampling %d documents per language (requested %d, min available %d)",
                target, sample_size, min_available)

    sampled = []
    for lang in languages:
        pool = buckets[lang]
        if len(pool) > target:
            random.seed(42)
            pool = random.sample(pool, target)
        sampled.extend(pool)
        logger.info("  %s: %d → %d (survival %.1f%%)",
                     lang, len(buckets[lang]), len(pool),
                     100.0 * len(pool) / max(len(buckets[lang]), 1))

    n = write_jsonl(output_path, sampled)
    logger.info("Wrote %d records to %s", n, output_path)
    return output_path
