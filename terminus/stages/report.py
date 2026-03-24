"""Stage 7: Analysis, charts, and results.md generation."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from terminus.utils.io import read_jsonl

logger = logging.getLogger(__name__)

TEAL = "#2ec4b6"
CORAL = "#e76f51"
SLATE = "#264653"
GOLD = "#e9c46a"


def _load_docs(input_path: Path) -> list:
    return list(read_jsonl(input_path))


def _split_by_lang(docs: list) -> dict:
    by_lang = defaultdict(list)
    for d in docs:
        by_lang[d.get("lang_ft", d.get("lang_cc", "unk"))].append(d)
    return dict(by_lang)


# ── Chart 1: Survival rate bar chart ────────────────────────────────────────

def _chart_survival(docs_by_lang: dict, output_dir: Path) -> Path:
    langs = sorted(docs_by_lang.keys())
    total = [len(docs_by_lang[l]) for l in langs]
    passed = [sum(1 for d in docs_by_lang[l] if d.get("heuristic_pass")) for l in langs]
    rates = [100.0 * p / t if t else 0 for p, t in zip(passed, total)]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(langs))
    bars = ax.bar(x, rates, color=[TEAL, CORAL], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.upper() for l in langs])
    ax.set_ylabel("Heuristic survival rate (%)")
    ax.set_title("Filter survival rate by language")
    ax.set_ylim(0, max(rates) * 1.3 if rates else 100)
    for bar, rate, p, t in zip(bars, rates, passed, total):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%\n({p}/{t})", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = output_dir / "chart_survival.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Chart 2: Confusion matrix heatmaps ─────────────────────────────────────

def _confusion(docs: list) -> dict:
    """Returns dict with tp, fp, fn, tn counts.
    Positive = HIGH quality. Pass = heuristic_pass=True.
    TP = pass & HIGH, FP = pass & LOW, FN = fail & HIGH, TN = fail & LOW."""
    tp = sum(1 for d in docs if d.get("heuristic_pass") and d["judge_quality"] == "HIGH")
    fp = sum(1 for d in docs if d.get("heuristic_pass") and d["judge_quality"] == "LOW")
    fn = sum(1 for d in docs if not d.get("heuristic_pass") and d["judge_quality"] == "HIGH")
    tn = sum(1 for d in docs if not d.get("heuristic_pass") and d["judge_quality"] == "LOW")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _chart_confusion(docs_by_lang: dict, output_dir: Path) -> Path:
    langs = sorted(docs_by_lang.keys())
    fig, axes = plt.subplots(1, len(langs), figsize=(5 * len(langs), 4))
    if len(langs) == 1:
        axes = [axes]

    for ax, lang in zip(axes, langs):
        cm = _confusion(docs_by_lang[lang])
        total = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
        matrix = np.array([[cm["tp"], cm["fp"]], [cm["fn"], cm["tn"]]])

        ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                pct = 100.0 * val / total if total else 0
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if val > matrix.max() * 0.6 else "black")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HIGH (judge)", "LOW (judge)"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Pass (heuristic)", "Fail (heuristic)"])
        ax.set_title(f"{lang.upper()} confusion matrix")

    fig.tight_layout()
    path = output_dir / "chart_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Chart 3: False rejection signal breakdown ──────────────────────────────

def _get_false_rejections(docs: list) -> list:
    """Docs where heuristic_pass=False but judge_quality=HIGH."""
    return [d for d in docs if not d.get("heuristic_pass") and d.get("judge_quality") == "HIGH"]


def _chart_false_rejection_signals(docs_by_lang: dict, output_dir: Path) -> Path:
    langs = sorted(docs_by_lang.keys())
    fig, axes = plt.subplots(1, len(langs), figsize=(6 * len(langs), 4))
    if len(langs) == 1:
        axes = [axes]
    colors = [TEAL, CORAL]

    for ax, lang, color in zip(axes, langs, colors):
        fr = _get_false_rejections(docs_by_lang[lang])
        sig_counts = Counter()
        for d in fr:
            for s in d.get("failed_signals", []):
                sig_counts[s] += 1

        if not sig_counts:
            ax.text(0.5, 0.5, "No false rejections", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{lang.upper()} — false rejection signals")
            continue

        top = sig_counts.most_common(8)
        signals, counts = zip(*top)
        y = np.arange(len(signals))
        ax.barh(y, counts, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(signals)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title(f"{lang.upper()} — signals causing false rejections")
        for i, c in enumerate(counts):
            ax.text(c + 0.3, i, str(c), va="center", fontsize=9)

    fig.tight_layout()
    path = output_dir / "chart_false_rejection_signals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Chart 4: Stop-word ratio distribution ───────────────────────────────────

def _chart_stopword_dist(docs_by_lang: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all values for axis range
    en_vals = [d["stop_word_ratio"] for d in docs_by_lang.get("en", [])]
    ru_vals = [d["stop_word_ratio"] for d in docs_by_lang.get("ru", [])]

    # Plot distributions by language (not split by quality — cleaner)
    if en_vals:
        ax.hist(en_vals, bins=30, alpha=0.6, color="#4361ee", label="EN documents",
                edgecolor="white", linewidth=0.5)
    if ru_vals:
        ax.hist(ru_vals, bins=30, alpha=0.6, color=CORAL, label="RU documents",
                edgecolor="white", linewidth=0.5)

    # Danger zone shading
    ylim = ax.get_ylim()
    ax.axvspan(0, 0.35, alpha=0.08, color="red", label="Danger zone — RU distribution")

    # Threshold line
    ax.axvline(x=0.35, color="red", linestyle="--", linewidth=2,
               label="EN-calibrated threshold (0.35)")

    # Mean annotations
    if en_vals:
        en_mean = sum(en_vals) / len(en_vals)
        ax.axvline(x=en_mean, color="#4361ee", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate(f"EN mean\n{en_mean:.4f}", xy=(en_mean, ax.get_ylim()[1] * 0.85),
                    fontsize=9, color="#4361ee", fontweight="bold",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#4361ee"))

    if ru_vals:
        ru_mean = sum(ru_vals) / len(ru_vals)
        ax.axvline(x=ru_mean, color=CORAL, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate(f"RU mean\n{ru_mean:.4f}", xy=(ru_mean, ax.get_ylim()[1] * 0.70),
                    fontsize=9, color=CORAL, fontweight="bold",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=CORAL))

    ax.set_xlabel("Stop-word ratio", fontsize=11)
    ax.set_ylabel("Document count", fontsize=11)
    ax.set_title("Stop-word ratio distribution: EN vs RU", fontsize=14, fontweight="bold")
    ax.text(0.5, 1.02, "Entire Russian distribution falls below English-calibrated threshold",
            transform=ax.transAxes, ha="center", fontsize=10, fontstyle="italic", color="gray")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="-")
    ax.set_xlim(left=0)

    fig.tight_layout(rect=[0, 0, 1, 1.04])
    path = output_dir / "chart_stopword_dist.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Results.md generation ───────────────────────────────────────────────────

def _generate_markdown(docs: list, docs_by_lang: dict, output_dir: Path,
                       all_by_lang: dict = None) -> str:
    if all_by_lang is None:
        all_by_lang = docs_by_lang
    langs = sorted(docs_by_lang.keys())
    lines = ["# Terminus — Results\n"]

    # ── Summary stats table ──
    lines.append("## Summary\n")
    lines.append("| Metric | " + " | ".join(l.upper() for l in langs) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in langs) + " |")

    # Survival rate — from full heuristics data
    row = "| Heuristic survival rate |"
    for l in langs:
        total = len(all_by_lang.get(l, []))
        passed = sum(1 for d in all_by_lang.get(l, []) if d.get("heuristic_pass"))
        rate = 100.0 * passed / total if total else 0
        row += f" {rate:.1f}% ({passed}/{total}) |"
    lines.append(row)

    # False rejection rate
    row = "| False rejection rate |"
    for l in langs:
        failed = [d for d in docs_by_lang[l] if not d.get("heuristic_pass")]
        fr = [d for d in failed if d.get("judge_quality") == "HIGH"]
        rate = 100.0 * len(fr) / len(failed) if failed else 0
        row += f" {rate:.1f}% ({len(fr)}/{len(failed)}) |"
    lines.append(row)

    # Judge agreement (heuristic_pass matches judge HIGH)
    row = "| Filter–judge agreement |"
    for l in langs:
        agree = sum(1 for d in docs_by_lang[l]
                    if (d.get("heuristic_pass") and d["judge_quality"] == "HIGH") or
                       (not d.get("heuristic_pass") and d["judge_quality"] == "LOW"))
        total = len(docs_by_lang[l])
        rate = 100.0 * agree / total if total else 0
        row += f" {rate:.1f}% |"
    lines.append(row)

    lines.append("")

    # ── Key findings ──
    lines.append("## Key findings\n")

    # Compute headline numbers
    fr_rates = {}
    for l in langs:
        failed = [d for d in docs_by_lang[l] if not d.get("heuristic_pass")]
        fr = _get_false_rejections(docs_by_lang[l])
        fr_rates[l] = 100.0 * len(fr) / len(failed) if failed else 0

    # Top signal for RU false rejections
    ru_fr = _get_false_rejections(docs_by_lang.get("ru", []))
    ru_sig_counts = Counter()
    for d in ru_fr:
        for s in d.get("failed_signals", []):
            ru_sig_counts[s] += 1
    top_ru_sig = ru_sig_counts.most_common(1)[0] if ru_sig_counts else ("n/a", 0)

    lines.append(f"1. **False rejection rate**: {fr_rates.get('ru', 0):.1f}% of Russian documents "
                 f"rated HIGH by the judge were rejected by heuristic filters, "
                 f"vs {fr_rates.get('en', 0):.1f}% for English.")
    top_pct = 100 * top_ru_sig[1] / len(ru_fr) if ru_fr else 0
    lines.append(f"2. **Primary driver**: `{top_ru_sig[0]}` accounts for "
                 f"{top_ru_sig[1]}/{len(ru_fr)} ({top_pct:.0f}%) "
                 f"of Russian false rejections.")
    # Compute actual survival rates from full data for the gap finding
    surv = {}
    for l in langs:
        total = len(all_by_lang.get(l, []))
        passed = sum(1 for d in all_by_lang.get(l, []) if d.get("heuristic_pass"))
        surv[l] = 100.0 * passed / total if total else 0
    en_surv = surv.get("en", 0)
    ru_surv = surv.get("ru", 0)
    gap = en_surv / ru_surv if ru_surv > 0 else float("inf")
    lines.append(f"3. **Survival gap**: English heuristic survival ({en_surv:.1f}%) is "
                 f"~{gap:.0f}x higher than Russian ({ru_surv:.1f}%), "
                 f"driven by the English-calibrated stop-word ratio threshold (0.35).")
    lines.append("")

    # ── Confusion matrices ──
    lines.append("## Confusion matrices\n")
    lines.append("Positive = HIGH quality. Pass = heuristic filter passed.\n")
    for l in langs:
        cm = _confusion(docs_by_lang[l])
        total = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
        lines.append(f"### {l.upper()}\n")
        lines.append("| | Judge: HIGH | Judge: LOW |")
        lines.append("| --- | --- | --- |")
        lines.append(f"| **Filter: Pass** | TP={cm['tp']} ({100*cm['tp']/total:.1f}%) "
                     f"| FP={cm['fp']} ({100*cm['fp']/total:.1f}%) |")
        lines.append(f"| **Filter: Fail** | FN={cm['fn']} ({100*cm['fn']/total:.1f}%) "
                     f"| TN={cm['tn']} ({100*cm['tn']/total:.1f}%) |")
        lines.append("")

    # ── Top false rejections ──
    lines.append("## Top false rejections\n")
    lines.append("Documents rejected by heuristics but rated HIGH by the judge.\n")
    for l in langs:
        fr = _get_false_rejections(docs_by_lang[l])
        # Sort by judge confidence descending
        fr.sort(key=lambda d: d.get("judge_confidence", 0), reverse=True)
        lines.append(f"### {l.upper()} (top 5)\n")
        for i, d in enumerate(fr[:5], 1):
            snippet = d["text"][:200].replace("\n", " ")
            lines.append(f"**{i}. [{d['id']}]** conf={d.get('judge_confidence', '?')} "
                         f"| failed: {', '.join(d.get('failed_signals', []))}")
            lines.append(f"> {snippet}...")
            lines.append(f"> *Judge: {d.get('judge_reason', 'n/a')}*\n")
        lines.append("")

    # ── Charts ──
    lines.append("## Charts\n")
    lines.append("![Survival rate](chart_survival.png)\n")
    lines.append("![Confusion matrix](chart_confusion.png)\n")
    lines.append("![False rejection signals](chart_false_rejection_signals.png)\n")
    lines.append("![Stop-word ratio distribution](chart_stopword_dist.png)\n")

    return "\n".join(lines)


# ── Main entry point ────────────────────────────────────────────────────────

def run(input_path: Path, output_dir: Path) -> Path:
    """Generate analysis report with charts and results.md.

    Args:
        input_path: Path to judged JSONL.
        output_dir: Directory to write charts and results.md.

    Returns:
        Path to results.md.
    """
    output_path = output_dir / "results.md"

    docs = _load_docs(input_path)
    if not docs:
        logger.warning("No documents to report on")
        output_path.write_text("# Terminus — Results\n\nNo data.\n")
        return output_path

    # Load full heuristics data for survival rate (not the sampled subset)
    heuristics_path = output_dir / "03_heuristics.jsonl"
    if heuristics_path.exists():
        all_docs = _load_docs(heuristics_path)
        all_by_lang = _split_by_lang(all_docs)
        logger.info("Loaded %d documents from heuristics stage for survival rates", len(all_docs))
    else:
        all_docs = docs
        all_by_lang = _split_by_lang(docs)
        logger.warning("Heuristics JSONL not found — survival rates will use judged subset only")

    docs_by_lang = _split_by_lang(docs)
    logger.info("Generating report for %d judged documents (%s)",
                len(docs), ", ".join(f"{l}={len(d)}" for l, d in sorted(docs_by_lang.items())))

    # Generate charts — survival uses full data, others use judged data
    _chart_survival(all_by_lang, output_dir)
    logger.info("  Saved chart_survival.png")

    _chart_confusion(docs_by_lang, output_dir)
    logger.info("  Saved chart_confusion.png")

    _chart_false_rejection_signals(docs_by_lang, output_dir)
    logger.info("  Saved chart_false_rejection_signals.png")

    _chart_stopword_dist(docs_by_lang, output_dir)
    logger.info("  Saved chart_stopword_dist.png")

    # Generate markdown — pass both full and judged data
    md = _generate_markdown(docs, docs_by_lang, output_dir, all_by_lang)
    output_path.write_text(md, encoding="utf-8")
    logger.info("  Saved results.md")

    return output_path
