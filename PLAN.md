# Terminus — Build Bible

> Multilingual web corpus curation for pre-training.  
> _The foundation of every great model is the data it was trained on._

---

## Thesis

### The claim

**Standard web corpus filters are English-first. Here's the cost.**

Most quality filters used in pre-training pipelines — heuristics, perplexity thresholds, quality classifiers — were calibrated on English CommonCrawl data. When applied naively to other languages, they systematically over-filter high-quality non-English text and under-filter low-quality content that happens to look "English-shaped."

### The mechanism (why it happens)

Stop-word ratio — one of the most common heuristic quality signals — is structurally invalid as a cross-lingual metric because morphological complexity inflates apparent lexical diversity in agglutinative and heavily inflected languages. A Russian document with the same semantic quality as an English one will score 30–50% lower on stop-word ratio purely due to grammatical structure.

Other failure modes:

- **Mean word length** thresholds calibrated on English penalise Russian (longer words due to morphology)
- **Punctuation ratio** thresholds break on Russian — «guillemets» vs standard quotes changes punctuation character counts
- **Stop-word ratio** conflates lexical frequency with grammatical function across typologically different languages

### The thesis in one sentence

_"English-calibrated stop-word and punctuation filters reject an estimated X% of high-quality Russian web text. The fix is a per-language stop-word list and two threshold offsets — a one-hour change that recovers meaningful training signal."_

### Three tiers of claims

**Tier 1 — Structural (reasoning, no experiment needed)**

- Stop-word ratio is linguistically invalid cross-lingually
- Mean word length thresholds systematically penalise Russian
- Punctuation ratio thresholds break on Russian quotation conventions

**Tier 2 — Empirical (what the pipeline shows)**

- Survival rate of RU vs EN documents under identical English-calibrated thresholds
- False rejection rate — RU documents rated HIGH by LLM judge but rejected by heuristics
- Which specific signals drive the most filter/judge disagreement

**Tier 3 — Prescriptive (what to do)**

- Language-aware threshold calibration closes X% of the gap
- A 50-word per-language stop-word list recovers most false rejections with minimal false passes

---

## Architecture

### Pipeline stages

```
WET file
    │
    ▼
[1. Ingest]          warcio parse → raw JSONL (id, url, lang_cc, text, char_count)
    │
    ▼
[2. Language filter] fasttext LID → keep EN + RU, sample to equal sizes (10k each)
    │
    ▼
[3. Heuristics]      compute all signals as floats, apply thresholds, log survival rate
    │
    ▼
[4. Dedup]           MinHash LSH per language, threshold=0.7, 5-gram signatures
    │
    ▼
[5. Stratified sample] 4 quadrants × ~75 docs = ~300 docs for judge
    │
    ▼
[6. LLM judge]       OpenRouter → Llama 3.1 8B bulk, Mistral Small for edge cases
    │
    ▼
[7. Report]          confusion matrix, survival rates, false rejection rate, charts, results.md
```

### Heuristic signals (all computed as floats, thresholded in config)

|Signal|Description|Cross-lingual issue|
|---|---|---|
|char_count|Total characters|None|
|word_count|Whitespace-split tokens|Minor|
|mean_word_length|Avg chars per word|Russian words are longer|
|punct_ratio|Punctuation / total chars|Russian quotation marks|
|digit_ratio|Digits / total chars|None|
|uppercase_ratio|Uppercase / total chars|None|
|stop_word_ratio|Stop words / total words|**Major — core thesis**|
|mean_line_length|Avg chars per line|None|
|bullet_ratio|Lines starting with bullets|None|
|ellipsis_ratio|Ellipsis patterns|None|

### Quadrant sampling for judge

```
                    heuristic_pass=True   heuristic_pass=False
lang=EN             Quadrant A            Quadrant B
lang=RU             Quadrant C            Quadrant D ← interesting
```

Quadrant D (RU, filter rejected) is where the thesis lives.  
Sample ~75 docs per quadrant, 300 total.

### CLI interface

```bash
terminus run \
  --wet-url https://data.commoncrawl.org/... \
  --languages en ru \
  --sample-size 10000 \
  --judge-model openrouter/meta-llama/llama-3.1-8b-instruct \
  --output ./results/run-001
```

---

## File structure

```
terminus/
├── README.md                    ← the artifact
├── PLAN.md                      ← this file
├── pyproject.toml
├── .env.example
│
├── terminus/
│   ├── __init__.py
│   ├── cli.py                   ← Click entry point
│   ├── config.py                ← dataclass config, loaded from yaml
│   │
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── ingest.py            ← WET download + warcio parse
│   │   ├── langfilter.py        ← fasttext LID + sampling
│   │   ├── heuristics.py        ← all filter signals
│   │   ├── dedup.py             ← MinHash LSH
│   │   ├── sampler.py           ← stratified quadrant sampling
│   │   ├── judge.py             ← OpenRouter API + retry logic
│   │   └── report.py            ← analysis + charts + results.md
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py                ← jsonl read/write helpers
│       └── text.py              ← tokenization, stop-word helpers
│
├── configs/
│   └── default.yaml             ← all thresholds and model names
│
├── results/                     ← gitignored
│   └── .gitkeep
│
├── notebooks/
│   └── analysis.ipynb           ← optional exploration
│
└── tests/
    ├── test_heuristics.py
    └── test_judge.py
```

---

## 24-hour schedule

|Hours|Task|Notes|
|---|---|---|
|0–1|Repo scaffold + WET download|Start download immediately, runs in background|
|1–3|ingest.py + langfilter.py|Verify EN/RU sample sizes|
|3–6|heuristics.py|Core stage — don't rush it|
|6–8|dedup.py + sampler.py|MinHash + quadrant sampling|
|**8**|**Checkpoint**|**Run end-to-end on 1k docs before touching judge**|
|8–12|judge.py|Test on 20 docs first, then full 300|
|12–15|Run full judge + report.py skeleton|Let judge run while you build report|
|15–18|report.py complete|Confusion matrix, charts, results.md|
|18–21|README|Real numbers, findings first|
|21–24|Buffer|Polish only — no new features|

### Rules for staying on track

- If a stage takes 2× longer than planned, cut scope — not sleep
- Near-dedup is cuttable if time is tight
- Judge on 150 docs is fine if 300 takes too long
- A clean repo with 3 working stages beats a broken 7-stage pipeline
- README matters as much as the code

---

## Claude Code prompts

Use these in order. Each stage reads from and writes to JSONL in the output directory.

---

### Prompt 1 — Scaffold

```
Create a Python package called terminus for multilingual web corpus quality 
filtering. Use pyproject.toml with these dependencies: warcio, fasttext, 
datasketch, openai (for OpenRouter), pandas, matplotlib, pyyaml, tqdm, click.

Create the full directory structure: terminus/stages/ with empty modules for 
ingest, langfilter, heuristics, dedup, sampler, judge, report. Create 
terminus/cli.py as the Click entry point with a 'run' command that accepts 
--wet-url, --languages, --sample-size, --judge-model, --output flags. Create 
configs/default.yaml with placeholder thresholds. Each stage should read from 
and write to JSONL files in the output directory.
```

---

### Prompt 2 — Ingest + language filter

```
Implement terminus/stages/ingest.py to download a CommonCrawl WET file from 
a URL using requests with streaming, parse it with warcio.ArchiveIterator, 
extract the URI, detected language header, and text content from each 
WARC-Type: conversion record. Write output as JSONL with fields: id, url, 
lang_cc (from CC header), text, char_count. Add a --dry-run flag that 
processes only the first 1000 records.

Then implement terminus/stages/langfilter.py to load the fasttext lid.176.bin 
model, run language detection on each document, keep only documents matching 
the target languages list, and sample to equal sizes per language. Log 
detection confidence distribution and survival rate per language.
```

---

### Prompt 3 — Heuristics

```
Implement terminus/stages/heuristics.py. For each document compute these 
signals as floats and store them all — do not threshold yet: char_count, 
word_count, mean_word_length, punct_ratio (punctuation chars / total chars), 
digit_ratio, uppercase_ratio, stop_word_ratio (for EN use nltk stopwords, 
for RU use the hardcoded list in terminus/utils/text.py), mean_line_length, 
bullet_ratio (lines starting with bullet patterns), ellipsis_ratio.

Load thresholds from the config yaml. Apply thresholds to produce a 
heuristic_pass boolean AND track which specific signal caused each rejection 
in a field called failed_signals (list). Keep all raw scores in the output 
JSONL. Log survival rate and top failure reasons per language at the end.

Use this stop-word ratio implementation:

def stopword_ratio(text: str, stopwords: set) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t.strip(".,!?;:\"'«»—-") in stopwords)
    return hits / len(tokens)
```

---

### Prompt 4 — Dedup + sampler

```
Implement terminus/stages/dedup.py using datasketch MinHashLSH with 
num_perm=128 and threshold=0.7. Run deduplication per language separately. 
Tokenize text into 5-grams for the MinHash signature. Log how many documents 
were removed per language.

Then implement terminus/stages/sampler.py to create a stratified sample for 
the judge stage. Divide documents into quadrants:
- Quadrant A: heuristic_pass=True, lang=en
- Quadrant B: heuristic_pass=False, lang=en  
- Quadrant C: heuristic_pass=True, lang=ru
- Quadrant D: heuristic_pass=False, lang=ru

Sample equally from each quadrant up to --sample-size total. Output a 
sampled JSONL preserving all score fields. Log quadrant sizes before and 
after sampling.
```

---

### Prompt 5 — Judge

```
Implement terminus/stages/judge.py using the OpenAI client pointed at 
OpenRouter (base_url=https://openrouter.ai/api/v1). Load the API key from 
the OPENROUTER_API_KEY environment variable.

For each document in the sample, truncate text to first 400 tokens 
(split on whitespace), send to the judge model with this system prompt:

---
You are a data quality judge for multilingual LLM pre-training corpora.
Your task is to assess whether a piece of web text is suitable for
training a large language model, regardless of its language or topic.

QUALITY CRITERIA — rate HIGH if the text:
- Is written by a human with coherent intent
- Contains real information, narrative, or argument a reader would find useful
- Has consistent grammar and natural sentence flow for its language
- Would be found on a legitimate website (news, blog, encyclopedia, forum, documentation)

Rate LOW if the text:
- Is spam, SEO keyword stuffing, or auto-generated filler
- Is boilerplate (cookie notices, navigation menus, error pages, legal disclaimers)
- Is garbled, truncated mid-sentence, or machine-translated poorly
- Contains primarily lists of links, product codes, or structured data with no prose
- Is duplicate or near-duplicate of itself within the passage

IMPORTANT: Do not penalize text for being on an unusual topic, being informal,
or being in a language other than English. A well-written Russian forum post
is HIGH quality. A poorly written English blog is LOW quality.

Respond with JSON only, no other text:
{"quality": "HIGH" or "LOW", "confidence": 0.0-1.0, "reason": "one sentence 
max 15 words", "primary_signal": "what most drove your decision"}
---

The user message template is:
Language: {lang}
Text (first 400 tokens):
{text}

Parse the JSON response. For documents where confidence < 0.5, add a 
needs_review=true flag. Add exponential backoff retry logic with max 3 
retries. Store judge_quality, judge_confidence, judge_reason, 
judge_primary_signal in output JSONL. Log cost estimate based on token counts.

Use openrouter/meta-llama/llama-3.1-8b-instruct as the primary model and openrouter/mistralai/mistral-small as the fallback for confidence < 0.5. The base URL is https://openrouter.ai/api/v1
```

---

### Prompt 6 — Report

```
Implement terminus/stages/report.py to generate a full analysis. Load the 
judge output JSONL. Compute and visualise:

1. Filter survival rate per language — bar chart (EN vs RU, side by side)

2. Confusion matrix of heuristic_pass vs judge_quality per language — 
   two 2×2 heatmaps side by side (EN and RU). Cells: TP, FP, FN, TN. 
   Annotate with counts and percentages.

3. False rejection rate per language — documents where heuristic_pass=False 
   but judge_quality=HIGH. This is the headline number.

4. Signal breakdown for false rejections — bar chart of which failed_signals 
   appear most in the false rejection set per language.

5. Stop-word ratio distribution — overlapping histograms for EN vs RU, 
   coloured by judge_quality (HIGH=teal, LOW=coral). This is the money chart.

Save all charts as PNG to the output directory. Generate results.md with:
- Summary stats table (survival rate, false rejection rate, judge agreement)
- Top 5 false rejections per language with text snippet (first 200 chars) 
  and judge_reason
- Top 3 key findings as bullet points
- Raw confusion matrix numbers
```

---

### Prompt 7 — README

```
Write a README.md for the Terminus project. Use this structure:

# Terminus
> Multilingual web corpus curation for pre-training.

## Findings  (FIRST — use {{PLACEHOLDER}} for real numbers)
## Quick start
## Pipeline  (ASCII art diagram of 7 stages)
## Judge design  (system prompt + model rationale)
## Heuristics  (table of all signals)
## Results  (embed chart filenames)
## Limitations  (honest — single WET segment, single language pair, 
                 judge not validated against human raters)
## Roadmap  (language-aware thresholds, more language pairs, Tessera)

Tone: direct, technical, like a well-written internal research note. 
No marketing language. Lead with findings, not features.
Tagline: "The foundation of every great model is the data it was trained on."
```

---

## Judge design

### System prompt (final version)

```
You are a data quality judge for multilingual LLM pre-training corpora.
Your task is to assess whether a piece of web text is suitable for
training a large language model, regardless of its language or topic.

QUALITY CRITERIA — rate HIGH if the text:
- Is written by a human with coherent intent
- Contains real information, narrative, or argument a reader would find useful
- Has consistent grammar and natural sentence flow for its language
- Would be found on a legitimate website (news, blog, encyclopedia, forum, documentation)

Rate LOW if the text:
- Is spam, SEO keyword stuffing, or auto-generated filler
- Is boilerplate (cookie notices, navigation menus, error pages, legal disclaimers repeated verbatim)
- Is garbled, truncated mid-sentence, or machine-translated poorly
- Contains primarily lists of links, product codes, or structured data with no prose
- Is duplicate or near-duplicate of itself within the passage

IMPORTANT: Do not penalize text for being on an unusual topic, being informal,
or being in a language other than English. A well-written Russian forum post
is HIGH quality. A poorly written English blog is LOW quality.

Respond with JSON only, no other text:
{"quality": "HIGH" or "LOW", "confidence": 0.0-1.0, "reason": "one sentence max 15 words", "primary_signal": "what most drove your decision"}
```

### User message template

```
Language: {lang}
Text (first 400 tokens):

{text}
```

### Model routing

|Case|Model|Reason|
|---|---|---|
|Bulk scoring|`meta-llama/llama-3.1-8b-instruct`|Cheap, fast, good enough for binary|
|confidence < 0.5|`mistralai/mistral-small`|Better reasoning on edge cases|
|Spot check|Any|Manual review|

Estimated cost: ~$2–5 for 300 documents

---

## Russian stop-word list

```python
STOPWORDS_RU = {
    # Pronouns
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "меня", "тебя", "его", "её", "нас", "вас", "их",
    "мне", "тебе", "ему", "ей", "нам", "вам", "им",
    "себя", "себе",

    # Demonstrative / determiners
    "это", "этот", "эта", "эти", "этого", "этой", "этим",
    "тот", "та", "те", "того", "той", "тем", "том",
    "такой", "такая", "такие", "такого",

    # Conjunctions
    "и", "а", "но", "или", "да", "что", "как", "если",
    "когда", "чтобы", "потому", "хотя", "либо", "ни",
    "зато", "однако", "причём", "притом",

    # Prepositions
    "в", "на", "с", "по", "к", "от", "до", "из",
    "за", "под", "над", "при", "про", "без", "для",
    "через", "между", "около", "после", "перед", "об",
    "со", "ко", "во",

    # Particles / fillers
    "не", "ни", "же", "бы", "ли", "вот", "ну", "уж",
    "ведь", "даже", "только", "уже", "ещё", "ещe",
    "тоже", "также", "именно", "просто", "вообще",

    # Common verbs (high frequency forms)
    "есть", "быть", "был", "была", "было", "были",
    "будет", "будут", "буду", "будем", "будете",
    "является", "являются", "нет", "нельзя", "можно",
    "надо", "нужно",

    # Adverbs
    "так", "там", "тут", "здесь", "где", "куда", "откуда",
    "когда", "всегда", "иногда", "теперь", "потом", "затем",
    "очень", "более", "менее", "совсем", "почти", "много",
    "мало", "сейчас", "раньше", "всего", "вдруг",

    # Quantifiers / misc
    "все", "всё", "всех", "всем", "один", "два", "три",
    "другой", "другие", "другого", "сам", "сама", "сами",
}
```

### Stop-word ratio implementation

```python
def stopword_ratio(text: str, stopwords: set) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t.strip(".,!?;:\"'«»—-") in stopwords)
    return hits / len(tokens)
```

### Calibration baseline

- Well-written English document: stop-word ratio ~0.45–0.55
- Well-written Russian document: stop-word ratio ~0.35–0.45 (structurally lower)
- If threshold is set at 0.35 (English-calibrated): cuts into the core Russian quality distribution1

---

## configs/default.yaml

```yaml
languages:
  - en
  - ru

sample_size: 10000

heuristics:
  thresholds:
    char_count_min: 200
    char_count_max: 100000
    word_count_min: 30
    mean_word_length_min: 3.0
    mean_word_length_max: 12.0
    punct_ratio_max: 0.15
    digit_ratio_max: 0.2
    uppercase_ratio_max: 0.2
    stop_word_ratio_min: 0.35   # ← this is the biased threshold
    mean_line_length_min: 20
    bullet_ratio_max: 0.5

dedup:
  num_perm: 128
  threshold: 0.7
  ngram_size: 5

judge:
  model: meta-llama/llama-3.1-8b-instruct
  fallback_model: mistralai/mistral-small
  confidence_threshold: 0.5
  max_tokens_input: 400
  max_retries: 3

sampling:
  judge_sample_size: 300
  per_quadrant: 75
```

---

## WET file download

Start immediately — takes 10–20 minutes:

```bash
wget "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/segments/1733066637688.46/wet/CC-MAIN-20241205095407-20241205125407-00000.wet.gz" \
  -O terminus_sample.wet.gz
```

Or browse available segments: https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-51/wet.paths.gz

---

## README structure (final shape)

```markdown
# Terminus
> tagline

## Findings          ← FIRST. Real numbers from your run.
## Quick start       ← pip install + one command
## Pipeline          ← ASCII diagram
## Judge design      ← prompt + model rationale
## Heuristics        ← signal table
## Results           ← embedded charts
## Limitations       ← honest
## Roadmap           ← language-aware thresholds, Tessera
```

### Key findings to highlight

1. False rejection rate: X% of RU docs rated HIGH by judge were rejected by heuristics
2. Primary driver: stop_word_ratio accounts for Y% of RU false rejections
3. EN false rejection rate for comparison: Z% (should be much lower)

---

## Reflection AI context

**Role:** Member of Technical Staff — Data Quality Engineer (Pre-training)  
**Company:** Frontier lab, ~$130M raised, open-weight superintelligence mission  
**Team from:** DeepMind, OpenAI, Google Brain, Meta, Anthropic  
**Key phrase from recruiter:** _"turning messy open-web data into high-quality corpora for training frontier models"_  
**Your angle:** Only candidate who can claim Aya + LLaMA 4 multilinguality + independent DQ toolkit work

### How to position Terminus

- "I built this to investigate a specific failure mode I'd noticed in multilingual data pipelines"
- "The false rejection rate number was the hypothesis — the pipeline was built to test it"
- "This is the seed of a larger open-source toolkit I'm calling Tessera"

---

_Last updated: March 2026_