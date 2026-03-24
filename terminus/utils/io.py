"""JSONL read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Union


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Union[Iterator[dict], list[dict]]) -> int:
    """Write records to a JSONL file. Returns count of records written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
