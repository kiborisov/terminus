"""Stage 3: Compute heuristic quality signals and apply thresholds."""

from __future__ import annotations

import logging
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

from terminus.utils.io import read_jsonl, write_jsonl
from terminus.utils.text import STOPWORDS_RU, stopword_ratio

logger = logging.getLogger(__name__)

# English stopwords — small self-contained set so we don't need nltk at runtime
STOPWORDS_EN = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
}

STOPWORDS = {"en": STOPWORDS_EN, "ru": STOPWORDS_RU}

PUNCT_CHARS = set(string.punctuation + "«»—–…„""''")
BULLET_RE = re.compile(r"^\s*(?:[-•*▪▸►→●◦‣⁃]|\d+[.)]\s|[a-zA-Z][.)]\s)")
ELLIPSIS_RE = re.compile(r"\.{3}|…")


def _compute_signals(text: str, lang: str) -> dict:
    """Compute all heuristic signals for a document."""
    chars = list(text)
    total_chars = len(chars)
    words = text.split()
    lines = text.split("\n")

    word_count = len(words)
    char_count = total_chars

    # Mean word length
    if words:
        mean_word_length = sum(len(w) for w in words) / word_count
    else:
        mean_word_length = 0.0

    # Punctuation ratio
    punct_count = sum(1 for c in chars if c in PUNCT_CHARS)
    punct_ratio = punct_count / total_chars if total_chars else 0.0

    # Digit ratio
    digit_count = sum(1 for c in chars if c.isdigit())
    digit_ratio = digit_count / total_chars if total_chars else 0.0

    # Uppercase ratio
    upper_count = sum(1 for c in chars if c.isupper())
    uppercase_ratio = upper_count / total_chars if total_chars else 0.0

    # Stop-word ratio
    sw_set = STOPWORDS.get(lang, STOPWORDS_EN)
    sw_ratio = stopword_ratio(text, sw_set)

    # Mean line length
    non_empty_lines = [l for l in lines if l.strip()]
    if non_empty_lines:
        mean_line_length = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
    else:
        mean_line_length = 0.0

    # Bullet ratio
    if non_empty_lines:
        bullet_count = sum(1 for l in non_empty_lines if BULLET_RE.match(l))
        bullet_ratio = bullet_count / len(non_empty_lines)
    else:
        bullet_ratio = 0.0

    # Ellipsis ratio (ellipsis patterns per 1000 chars)
    ellipsis_count = len(ELLIPSIS_RE.findall(text))
    ellipsis_ratio = (ellipsis_count / total_chars * 1000) if total_chars else 0.0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "mean_word_length": round(mean_word_length, 4),
        "punct_ratio": round(punct_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "uppercase_ratio": round(uppercase_ratio, 4),
        "stop_word_ratio": round(sw_ratio, 4),
        "mean_line_length": round(mean_line_length, 2),
        "bullet_ratio": round(bullet_ratio, 4),
        "ellipsis_ratio": round(ellipsis_ratio, 4),
    }


def _apply_thresholds(signals: dict, thresholds: dict) -> List[str]:
    """Check signals against thresholds. Return list of failed signal names."""
    failed = []

    def _check_min(key, threshold_key):
        if threshold_key in thresholds and signals[key] < thresholds[threshold_key]:
            failed.append(key)

    def _check_max(key, threshold_key):
        if threshold_key in thresholds and signals[key] > thresholds[threshold_key]:
            failed.append(key)

    _check_min("char_count", "char_count_min")
    _check_max("char_count", "char_count_max")
    _check_min("word_count", "word_count_min")
    _check_min("mean_word_length", "mean_word_length_min")
    _check_max("mean_word_length", "mean_word_length_max")
    _check_max("punct_ratio", "punct_ratio_max")
    _check_max("digit_ratio", "digit_ratio_max")
    _check_max("uppercase_ratio", "uppercase_ratio_max")
    _check_min("stop_word_ratio", "stop_word_ratio_min")
    _check_min("mean_line_length", "mean_line_length_min")
    _check_max("bullet_ratio", "bullet_ratio_max")

    return failed


def run(input_path: Path, output_dir: Path, config: dict) -> Path:
    """Compute heuristic signals and threshold them.

    Args:
        input_path: Path to langfilter JSONL.
        output_dir: Directory to write output JSONL.
        config: Heuristics config dict with 'thresholds' key.

    Returns:
        Path to the output JSONL file.
    """
    output_path = output_dir / "03_heuristics.jsonl"
    thresholds = config.get("thresholds", {})

    # Per-language counters
    total_by_lang = Counter()
    pass_by_lang = Counter()
    fail_reasons_by_lang = defaultdict(Counter)

    def process():
        for doc in read_jsonl(input_path):
            lang = doc.get("lang_ft", doc.get("lang_cc", "en"))
            signals = _compute_signals(doc["text"], lang)
            failed = _apply_thresholds(signals, thresholds)

            doc.update(signals)
            doc["heuristic_pass"] = len(failed) == 0
            doc["failed_signals"] = failed

            total_by_lang[lang] += 1
            if doc["heuristic_pass"]:
                pass_by_lang[lang] += 1
            for sig in failed:
                fail_reasons_by_lang[lang][sig] += 1

            yield doc

    n = write_jsonl(output_path, process())

    # Log results
    logger.info("Heuristic filtering complete — %d documents", n)
    for lang in sorted(total_by_lang):
        total = total_by_lang[lang]
        passed = pass_by_lang[lang]
        rate = 100.0 * passed / total if total else 0.0
        logger.info("  %s: %d / %d passed (%.1f%% survival)", lang, passed, total, rate)

        top_reasons = fail_reasons_by_lang[lang].most_common(5)
        if top_reasons:
            reasons_str = ", ".join(f"{sig}={cnt}" for sig, cnt in top_reasons)
            logger.info("  %s top failures: %s", lang, reasons_str)

    return output_path
