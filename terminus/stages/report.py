"""Stage 7: Analysis, charts, and results.md generation."""

from pathlib import Path


def run(input_path: Path, output_dir: Path) -> Path:
    """Generate analysis report with charts and results.md.

    Args:
        input_path: Path to judged JSONL.
        output_dir: Directory to write charts and results.md.

    Returns:
        Path to results.md.
    """
    output_path = output_dir / "results.md"
    # TODO: implement
    return output_path
