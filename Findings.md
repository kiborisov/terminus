real numbers as they come in, updated live

## Quantified findings (dry-run, 101 docs per language)

### Headline numbers
- EN survival rate: 6.9%
- RU survival rate: 0.0% — complete wipeout
- Primary driver: stop_word_ratio (100% of RU failures)

### The mechanism, quantified
- Mean stop_word_ratio: EN 0.2033 vs RU 0.1429 (~30% lower structurally)
- RU distribution range: 0.00–0.3231
- English-calibrated threshold: 0.35
- Gap: the ENTIRE Russian distribution sits below the threshold
- The threshold isn't biased — it's categorically unreachable for Russian

### What the judge will tell us
How many of those 0% surviving RU docs are actually HIGH quality?
That's the false rejection rate — the cost of this bias.
```

The phrase **"categorically unreachable"** is the one to use in your README and in conversation with Reflection. It's precise and striking.

Now — the dry-run sample is 101 docs per language which is enough to prove the mechanism but thin for the judge stage. Before moving to Prompt 4, run the full pipeline on the complete `sample.wet.gz` without dry-run to get a bigger sample. Tell Claude Code:
```
Run stages 1 and 2 on sample.wet.gz without dry-run, 
targeting 5000 docs per language, output to results/run-002/




## Quantified findings (full run, 1,331 docs per language)

### Headline
- EN survival rate: 9.4% (125/1,331)
- RU survival rate: 0.2% (3/1,331)
- Ratio: English survives at 47x the rate of Russian
- Primary driver: stop_word_ratio (99.7% of RU failures)

### The mechanism
- stop_word_ratio threshold of 0.35 fails 99.7% of Russian docs
- Only 4 RU documents out of 1,331 cleared that single threshold
- The threshold isn't biased — it's categorically unreachable for Russian

### Russian underrepresentation compounds the problem
- 19,637 total WET records → only 1,331 RU (6.8%)
- Less data to begin with, then near-total filtering on top
- Combined effect: Russian nearly disappears from the training corpus