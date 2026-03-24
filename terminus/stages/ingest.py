"""Stage 1: Download and parse CommonCrawl WET files into JSONL."""

from __future__ import annotations

import hashlib
import logging
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

from terminus.utils.io import write_jsonl

logger = logging.getLogger(__name__)

# Map ISO 639-3 (used in CC headers) to ISO 639-1 (used everywhere else)
LANG3_TO_LANG2 = {
    "eng": "en", "rus": "ru", "zho": "zh", "deu": "de", "fra": "fr",
    "spa": "es", "por": "pt", "ita": "it", "jpn": "ja", "kor": "ko",
    "ara": "ar", "hin": "hi", "tur": "tr", "pol": "pl", "nld": "nl",
    "ukr": "uk", "ces": "cs", "ron": "ro", "hun": "hu", "swe": "sv",
}


def _parse_cc_lang(header: str) -> str:
    """Extract the primary language from the CC header, mapped to ISO 639-1."""
    if not header:
        return "unk"
    primary = header.split(",")[0].strip()
    return LANG3_TO_LANG2.get(primary, primary)


def _open_wet(wet_url: str) -> BytesIO:
    """Stream a WET file from a URL or open a local file."""
    if wet_url.startswith(("http://", "https://")):
        logger.info("Downloading %s ...", wet_url)
        resp = requests.get(wet_url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        buf = BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading WET") as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                buf.write(chunk)
                pbar.update(len(chunk))
        buf.seek(0)
        return buf
    else:
        # Local file path
        local = Path(wet_url).expanduser()
        logger.info("Reading local file %s", local)
        return BytesIO(local.read_bytes())


def run(wet_url: str, output_dir: Path, dry_run: bool = False) -> Path:
    """Download a WET file and extract records to JSONL.

    Args:
        wet_url: URL or local path to a CommonCrawl WET file (.wet.gz).
        output_dir: Directory to write output JSONL.
        dry_run: If True, process only first 1000 records.

    Returns:
        Path to the output JSONL file.
    """
    output_path = output_dir / "01_ingest.jsonl"
    max_records = 1000 if dry_run else None

    raw = _open_wet(wet_url)

    def records():
        count = 0
        skipped = 0
        for record in ArchiveIterator(raw):
            if record.rec_type != "conversion":
                continue

            text = record.content_stream().read().decode("utf-8", errors="replace").strip()
            if not text:
                skipped += 1
                continue

            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            lang_header = record.rec_headers.get_header("WARC-Identified-Content-Language") or ""
            record_id = record.rec_headers.get_header("WARC-Record-ID") or ""

            # Deterministic short ID from the WARC record ID
            doc_id = hashlib.md5(record_id.encode()).hexdigest()[:12]

            yield {
                "id": doc_id,
                "url": url,
                "lang_cc": _parse_cc_lang(lang_header),
                "text": text,
                "char_count": len(text),
            }

            count += 1
            if max_records and count >= max_records:
                logger.info("Dry-run limit reached (%d records)", max_records)
                break

        logger.info("Ingested %d records (%d empty skipped)", count, skipped)

    n = write_jsonl(output_path, records())
    logger.info("Wrote %d records to %s", n, output_path)
    return output_path
