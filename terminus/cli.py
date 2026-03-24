"""CLI entry point for Terminus."""

import logging
from pathlib import Path

import click

from terminus.config import load_config


@click.group()
def cli():
    """Terminus — Multilingual web corpus quality filtering."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option(
    "--wet-url",
    required=True,
    help="URL to a CommonCrawl WET file (.wet.gz).",
)
@click.option(
    "--languages",
    default="en,ru",
    show_default=True,
    help="Comma-separated ISO 639-1 language codes.",
)
@click.option(
    "--sample-size",
    default=10000,
    show_default=True,
    type=int,
    help="Target documents per language after language filtering.",
)
@click.option(
    "--judge-model",
    default=None,
    help="OpenRouter model ID for judge (overrides config).",
)
@click.option(
    "--output",
    "output_dir",
    default="./results/run-001",
    show_default=True,
    type=click.Path(),
    help="Output directory for JSONL files and reports.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to config YAML (defaults to configs/default.yaml).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Process only the first 1000 records.",
)
@click.option(
    "--test",
    is_flag=True,
    default=False,
    help="Judge only the first 5 documents and print full responses.",
)
def run(wet_url, languages, sample_size, judge_model, output_dir, config_path, dry_run, test):
    """Run the full Terminus pipeline."""
    from terminus.stages import ingest, langfilter, heuristics, dedup, sampler, judge, report

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(config_path) if config_path else None)
    langs = [l.strip() for l in languages.split(",")]

    if judge_model:
        config["judge"]["model"] = judge_model

    click.echo(f"=== Terminus pipeline ===")
    click.echo(f"WET URL:    {wet_url}")
    click.echo(f"Languages:  {langs}")
    click.echo(f"Sample:     {sample_size} per language")
    click.echo(f"Output:     {output}")
    click.echo()

    # Stage 1: Ingest
    click.echo("[1/7] Ingesting WET file...")
    ingest_path = ingest.run(wet_url, output, dry_run=dry_run)

    # Stage 2: Language filter
    click.echo("[2/7] Filtering by language...")
    langfilter_path = langfilter.run(ingest_path, output, langs, sample_size)

    # Stage 3: Heuristics
    click.echo("[3/7] Computing heuristic signals...")
    heuristics_path = heuristics.run(langfilter_path, output, config["heuristics"])

    # Stage 4: Dedup
    click.echo("[4/7] Deduplicating...")
    dedup_path = dedup.run(heuristics_path, output, config["dedup"])

    # Stage 5: Sampler
    click.echo("[5/7] Stratified sampling...")
    sampler_path = sampler.run(dedup_path, output, config["sampling"])

    # Stage 6: Judge
    click.echo("[6/7] Running LLM judge...")
    judge_path = judge.run(sampler_path, output, config["judge"], test=test)

    # Stage 7: Report
    click.echo("[7/7] Generating report...")
    report_path = report.run(judge_path, output)

    click.echo()
    click.echo(f"Done. Report at {report_path}")


if __name__ == "__main__":
    cli()
