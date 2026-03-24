# Terminus

> Multilingual web corpus curation for pre-training.
> *The foundation of every great model is the data it was trained on.*

## Findings

Standard web corpus filters are English-first. Here's what that costs.

I ran a 7-stage pipeline on a single CommonCrawl WET segment (CC-MAIN-2024-51), applying identical English-calibrated heuristic filters to both EN and RU documents, then scoring a stratified sample with an LLM quality judge.

| Metric | EN | RU |
| --- | --- | --- |
| Heuristic survival rate | 9.4% (125/1,331) | 0.2% (3/1,331) |
| False rejection rate | 13.3% | 40.0% |
| Stop-word ratio failures | 89.2% of rejections | 99.7% of rejections |

**The headline numbers:**

- **47x survival gap.** English documents pass heuristic filters at 9.4%, Russian at 0.2%.
- **3x false rejection gap.** 40% of Russian documents the judge rated HIGH were rejected by heuristics, vs 13.3% for English.
- **One threshold causes it all.** 99.7% of Russian failures trace to a single signal: `stop_word_ratio` with an English-calibrated minimum of 0.35.
- **Categorically unreachable.** The entire Russian stop-word ratio distribution (max 0.3231, mean 0.1429) sits below the 0.35 threshold. No Russian document can pass, regardless of quality.
- **Estimated impact.** At this rejection rate, roughly 530 high-quality Russian documents are discarded per WET segment. Across the full CC-MAIN-2024-51 crawl (94,000+ segments), this scales to tens of millions.

**Why it happens:** Morphological complexity in Russian (agglutination, inflection, case marking) means fewer tokens match a whitespace-split stop-word list. A well-written Russian document scores 0.35-0.45 on stop-word ratio; a well-written English document scores 0.45-0.55. The threshold was calibrated for English and never adjusted.

![Stop-word ratio distribution](results/run-002/chart_stopword_dist.png)

## Quick start

```bash
git clone https://github.com/yourname/terminus.git
cd terminus
pip install -e .
```

Download a WET file:

```bash
wget "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/segments/1733066637688.46/wet/CC-MAIN-20241205095407-20241205125407-00000.wet.gz" \
  -O sample.wet.gz
```

Create `.env` with your OpenRouter API key:

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

Run the full pipeline:

```bash
terminus run \
  --wet-url ./sample.wet.gz \
  --languages en,ru \
  --sample-size 10000 \
  --output ./results/run-001
```

Test the judge on 5 documents first:

```bash
terminus run \
  --wet-url ./sample.wet.gz \
  --output ./results/test \
  --dry-run \
  --test
```

## Pipeline

```
WET file (60MB compressed, ~20k documents)
    |
    v
[1. Ingest]           warcio parse -> raw JSONL (id, url, lang_cc, text, char_count)
    |
    v
[2. Language filter]  FastText LID -> keep target languages, balance to equal sizes
    |
    v
[3. Heuristics]       10 signals as floats, threshold from config, track failed_signals
    |
    v
[4. Dedup]            MinHash LSH per language (128 perms, threshold=0.7, 5-grams)
    |
    v
[5. Stratified sample]  4 quadrants (lang x heuristic_pass) -> ~300 docs for judge
    |
    v
[6. LLM judge]        OpenRouter API, Llama 3.1 8B bulk + Mistral Small fallback
    |
    v
[7. Report]           Confusion matrix, survival rates, charts, results.md
```

Each stage reads from and writes to JSONL files in the output directory. Stages are independently re-runnable.

## Judge design

The judge scores each document as HIGH or LOW quality using a language-agnostic rubric.

**Primary model:** `meta-llama/llama-3.1-8b-instruct` via OpenRouter. Fast, cheap (~$0.01 for 300 docs), good enough for binary quality classification.

**Fallback model:** `mistralai/mistral-small` for documents where primary confidence < 0.5. Better reasoning on edge cases.

**System prompt (abbreviated):**

```
You are a data quality judge for multilingual LLM pre-training corpora.

Rate HIGH if: written by a human, coherent intent, real information,
consistent grammar, legitimate website content.

Rate LOW if: spam, boilerplate, garbled, keyword stuffing, structured
data without prose.

IMPORTANT: Do not penalize text for being in a language other than English.
A well-written Russian forum post is HIGH quality.

Respond with JSON: {"quality", "confidence", "reason", "primary_signal"}
```

**Quadrant sampling** ensures the judge sees documents from all four combinations of (language x filter outcome), not just the ones that passed. This is what makes the false rejection rate measurable.

```
                    heuristic_pass=True   heuristic_pass=False
lang=EN             Quadrant A            Quadrant B
lang=RU             Quadrant C            Quadrant D  <-- where the thesis lives
```

## Heuristics

All signals are computed as floats and stored in the output JSONL. Thresholds are applied separately, and each rejection records which signal caused it.

| Signal | Description | Threshold | Cross-lingual issue |
| --- | --- | --- | --- |
| char_count | Total characters | 200 - 100,000 | None |
| word_count | Whitespace-split tokens | min 30 | Minor |
| mean_word_length | Avg chars per word | 3.0 - 12.0 | Russian words are longer |
| punct_ratio | Punctuation / total chars | max 0.15 | Russian guillemets |
| digit_ratio | Digits / total chars | max 0.2 | None |
| uppercase_ratio | Uppercase / total chars | max 0.2 | None |
| **stop_word_ratio** | **Stop words / total words** | **min 0.35** | **Core thesis** |
| mean_line_length | Avg chars per line | min 20 | None |
| bullet_ratio | Lines starting with bullets | max 0.5 | None |
| ellipsis_ratio | Ellipsis patterns per 1k chars | (not thresholded) | None |

The `stop_word_ratio` threshold of 0.35 is deliberately set to the English-calibrated value used in standard pipelines. This is the biased threshold the project exists to expose.

## Results

![Survival rate by language](results/run-002/chart_survival.png)

![Confusion matrix](results/run-002/chart_confusion.png)

![False rejection signals](results/run-002/chart_false_rejection_signals.png)

![Stop-word ratio distribution](results/run-002/chart_stopword_dist.png)

Full results with confusion matrices, top false rejections, and judge reasoning: [`results/run-002/results.md`](results/run-002/results.md)

## Limitations

This is a pilot run. Be skeptical of the exact numbers.

- **Single WET segment.** 60MB of 5.63TiB total in CC-MAIN-2024-51. One segment is not representative of the full crawl.
- **Small judge sample.** 228 documents scored (75 per quadrant, except Quadrant C with only 3). Statistical power is limited.
- **LLM judge not validated.** No human rater baseline. The judge (Llama 3.1 8B) may have its own biases, particularly on short or domain-specific Russian text.
- **Single language pair.** EN/RU only. The thesis likely applies to other morphologically rich languages (Turkish, Finnish, Hungarian, Arabic) but this hasn't been tested.
- **No boilerplate stripping.** WET files contain extracted text with navigation, menus, and footers intact. A production pipeline would strip these first.

## Roadmap

- **Scale validation.** Run on the full CC-MAIN-2024-51 crawl (94,000+ segments) to confirm the false rejection rate at corpus scale.
- **Language-aware thresholds.** The proposed fix: per-language stop-word lists and calibrated threshold offsets. A one-hour change that should recover most false rejections.
- **More language pairs.** Expand to Turkish, Finnish, Arabic, Hindi, Japanese to test generality across typological families.
- **Human validation.** Annotate 200+ documents to benchmark the LLM judge against human raters.
- **Terminus.** Release as an open-source multilingual data quality toolkit with pluggable filters, language-aware defaults, and built-in bias detection.

---

Built with CommonCrawl data. Judge powered by OpenRouter.
