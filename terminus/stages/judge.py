"""Stage 6: LLM quality judge via OpenRouter."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from tqdm import tqdm

from terminus.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a data quality judge for multilingual LLM pre-training corpora.
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
{"quality": "HIGH" or "LOW", "confidence": 0.0-1.0, "reason": "one sentence max 15 words", "primary_signal": "what most drove your decision"}"""

USER_TEMPLATE = "Language: {lang}\nText (first 400 tokens):\n{text}"


def _truncate(text: str, max_tokens: int = 400) -> str:
    """Truncate text to first max_tokens whitespace-split tokens."""
    tokens = text.split()[:max_tokens]
    return " ".join(tokens)


def _parse_judge_response(content: str) -> dict:
    """Parse the JSON response from the judge model."""
    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    data = json.loads(content)
    return {
        "judge_quality": data.get("quality", "UNKNOWN").upper(),
        "judge_confidence": float(data.get("confidence", 0.0)),
        "judge_reason": str(data.get("reason", "")),
        "judge_primary_signal": str(data.get("primary_signal", "")),
    }


def _call_judge(client: OpenAI, model: str, lang: str, text: str, max_retries: int) -> dict:
    """Call the judge model with exponential backoff retry."""
    user_msg = USER_TEMPLATE.format(lang=lang, text=_truncate(text))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            content = response.choices[0].message.content
            result = _parse_judge_response(content)

            # Track token usage
            usage = response.usage
            if usage:
                result["_prompt_tokens"] = usage.prompt_tokens
                result["_completion_tokens"] = usage.completion_tokens

            return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Parse error on attempt %d: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                return {
                    "judge_quality": "ERROR",
                    "judge_confidence": 0.0,
                    "judge_reason": f"parse_error: {e}",
                    "judge_primary_signal": "error",
                    "_prompt_tokens": 0,
                    "_completion_tokens": 0,
                }
        except Exception as e:
            wait = 2 ** (attempt + 1)
            logger.warning("API error on attempt %d: %s — retrying in %ds", attempt + 1, e, wait)
            if attempt == max_retries - 1:
                return {
                    "judge_quality": "ERROR",
                    "judge_confidence": 0.0,
                    "judge_reason": f"api_error: {e}",
                    "judge_primary_signal": "error",
                    "_prompt_tokens": 0,
                    "_completion_tokens": 0,
                }
            time.sleep(wait)

    # Should not reach here
    return {"judge_quality": "ERROR", "judge_confidence": 0.0,
            "judge_reason": "max_retries", "judge_primary_signal": "error",
            "_prompt_tokens": 0, "_completion_tokens": 0}


def run(input_path: Path, output_dir: Path, config: dict, test: bool = False) -> Path:
    """Score documents with an LLM judge via OpenRouter.

    Args:
        input_path: Path to sampled JSONL.
        output_dir: Directory to write output JSONL.
        config: Judge config dict (model, fallback_model, confidence_threshold, etc.).
        test: If True, process only first 5 docs and print full JSON responses.

    Returns:
        Path to the output JSONL file.
    """
    output_path = output_dir / "06_judged.jsonl"

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    primary_model = config.get("model", "meta-llama/llama-3.1-8b-instruct")
    fallback_model = config.get("fallback_model", "mistralai/mistral-small")
    confidence_threshold = config.get("confidence_threshold", 0.5)
    max_retries = config.get("max_retries", 3)

    docs = list(read_jsonl(input_path))
    if test:
        docs = docs[:5]
        logger.info("TEST MODE — judging first 5 documents only")
    logger.info("Judging %d documents with %s (fallback: %s)", len(docs), primary_model, fallback_model)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    results = []

    for doc in tqdm(docs, desc="Judging"):
        lang = doc.get("lang_ft", doc.get("lang_cc", "en"))

        # Primary model pass
        result = _call_judge(client, primary_model, lang, doc["text"], max_retries)

        # Fallback for low confidence
        needs_review = False
        if result["judge_confidence"] < confidence_threshold and result["judge_quality"] != "ERROR":
            needs_review = True
            logger.debug("Low confidence (%.2f) for %s — re-judging with %s",
                         result["judge_confidence"], doc["id"], fallback_model)
            fallback = _call_judge(client, fallback_model, lang, doc["text"], max_retries)
            if fallback["judge_quality"] != "ERROR":
                result = fallback
                result["judge_model"] = fallback_model
            else:
                result["judge_model"] = primary_model
        else:
            result["judge_model"] = primary_model

        # Accumulate token counts
        total_prompt_tokens += result.pop("_prompt_tokens", 0)
        total_completion_tokens += result.pop("_completion_tokens", 0)

        result["needs_review"] = needs_review
        doc.update(result)
        results.append(doc)

        if test:
            print(json.dumps({
                "id": doc["id"],
                "lang": lang,
                "quadrant": doc.get("quadrant", "?"),
                "text_preview": doc["text"][:150] + "...",
                "judge_quality": doc["judge_quality"],
                "judge_confidence": doc["judge_confidence"],
                "judge_reason": doc["judge_reason"],
                "judge_primary_signal": doc["judge_primary_signal"],
                "judge_model": doc["judge_model"],
                "needs_review": doc["needs_review"],
            }, indent=2, ensure_ascii=False))
            print()

    n = write_jsonl(output_path, results)

    # Cost estimate (OpenRouter Llama 3.1 8B: ~$0.06/M input, ~$0.06/M output)
    est_cost = (total_prompt_tokens * 0.06 + total_completion_tokens * 0.06) / 1_000_000
    logger.info("Judge complete — %d documents scored", n)
    logger.info("  Tokens: %d prompt + %d completion = %d total",
                total_prompt_tokens, total_completion_tokens,
                total_prompt_tokens + total_completion_tokens)
    logger.info("  Estimated cost: $%.4f", est_cost)

    # Quick breakdown
    high = sum(1 for d in results if d["judge_quality"] == "HIGH")
    low = sum(1 for d in results if d["judge_quality"] == "LOW")
    errors = sum(1 for d in results if d["judge_quality"] == "ERROR")
    logger.info("  HIGH: %d | LOW: %d | ERROR: %d", high, low, errors)

    return output_path
