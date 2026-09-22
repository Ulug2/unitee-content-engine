# UniTee Content Engine

Generates realistic, anonymous, diverse student posts for UniTee (an anonymous university
social app) using GPT-5-mini. The pipeline scrapes real anonymous student channels on Telegram
to learn a *statistical style reference* (tone, length, language mix), then generates entirely
original posts from that style — never copying, paraphrasing, or reusing real posts or real
people's information.

This repo holds the **code and config only**. The scraped data itself is intentionally not
committed — see [Data & privacy](#data--privacy) below.

## What it does

1. **Scrape** a couple of anonymous Telegram channels into `raw_telegram_posts.json`
   (`export_channels.py`).
2. **Clean** the scrape: strip contact info, identification requests, harassment, ads and noise,
   and split what's left into `safe_style_posts.json` (usable as a style reference),
   `review_posts.json` (borderline) and `excluded_posts.json` (rejected), each with the reason
   (`clean_dataset.py`).
3. **Analyze** `safe_style_posts.json` into `unitee_style_profile.json`: aggregate statistics
   only (length distribution, language mix, punctuation habits, topic spread) — no post text
   (`style_analyzer.py`).
4. **Generate** 20 new, original posts from that style profile (`generate_posts.py`), the main
   entry point:

   ```
   style / rules / strategy / real-post corpus
                 |
          Stage 1: Seeds            neutral "what happened / what people ask" notes,
                                     one call per topic domain, de-duplicated by meaning
                 |
          Stage 2: Generators       15 post types (question, confession, rant, story, ...),
                                     each with its own objective, shape and typical length
                 |
          Local safety / cleanup    regex + rule-based; no API calls
                 |
          Stage 3: AI scoring       quality + "formulaic" + "ai_feel" labels, batched
                 |
          Stage 4: Selection        quality + topic/type/language/length/opener coverage
                                     - redundancy, no fixed quotas
                 |
          Validation                hard errors for safety/duplicates, warnings for
                                     concentration (too many questions, repeated phrases, ...)
                 |
          generated_posts.json      JSON array of exactly 20 strings
   ```

   Diversity is built *during* generation (different generators, different underlying
   situations, different languages) rather than bolted on afterward with quotas. See the
   module docstring at the top of `generate_posts.py` for the full design rationale.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in real values
python generate_posts.py
```

Needs, as environment variables (a `.env` file is loaded automatically where scripts use it):

| Variable | Used by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `generate_posts.py` | GPT-5-mini calls for seeding, generation, scoring |
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | `export_channels.py`, `inspect_channels.py`, `test_telegram.py` | Telegram API credentials (from [my.telegram.org](https://my.telegram.org)) — only needed to re-scrape, not to generate posts |

`generate_posts.py` needs `safe_style_posts.json` on disk (see below) plus the three config
files (`unitee_style_profile.json`, `generation_rules.json`, `content_strategy.json`, all
included in this repo). It writes `generated_posts.json` (the final 20 posts, as a JSON array
of strings — the stable output contract other parts of the project rely on) and
`generation_debug.json` (the full candidate pool with scores/labels, for diagnosing a run; not
committed, regenerated every run).

## Data & privacy

**Not committed**, on purpose (`.gitignore`):

- `.env`, `*.session`, `telegram_login_qr.png` — credentials and login material.
- `raw_telegram_posts.json`, `cleaned_telegram_posts.json`, `safe_style_posts.json`,
  `flagged_telegram_posts.json`, `excluded_posts.json`, `review_posts.json` — the scraped
  Telegram data itself. These are real, non-consented posts from real students. Some contain
  names, identifying descriptions, or requests for someone's contact info — exactly the content
  this project's own rules (`generation_rules.json`) forbid generating. They stay local only,
  even though this is a private repo.
- `generation_debug.json` — a large per-run dump, regenerated every run.

**Committed**: everything downstream of the raw scrape that doesn't contain real post text —
`unitee_style_profile.json` (aggregate stats: length/language/punctuation distributions, no
post content) and the generated output itself, `generated_posts.json` (fully synthetic —
GPT-5-mini's original writing, not real posts).

To run this pipeline from a fresh clone, you need your own `raw_telegram_posts.json` →
`clean_dataset.py` → `safe_style_posts.json`, which requires Telegram API credentials and access
to the source channels. Generation-only work (`generate_posts.py`) needs `safe_style_posts.json`
specifically; everything else it reads is already in this repo.

## Files

| File | Role |
|---|---|
| `export_channels.py` | Scrapes configured Telegram channels into `raw_telegram_posts.json`. |
| `inspect_channels.py`, `inspect_style.py`, `analyze_dataset.py` | One-off inspection scripts used while building the pipeline. |
| `test_telegram.py` | Telegram login test / QR pairing helper. |
| `clean_dataset.py` | Filters the raw scrape into safe / review / excluded buckets. |
| `style_analyzer.py` | Turns `safe_style_posts.json` into `unitee_style_profile.json`. |
| `unitee_style_profile.json` | Aggregate style statistics used to steer generation. |
| `generation_rules.json` | Hard safety/factual rules (never invent university facts, never identify people, etc.). |
| `content_strategy.json` | Editorial targets (topic mix, tone, length ranges) referenced during generation. |
| `generate_posts.py` | The generator — see pipeline diagram above. |
| `generated_posts.json` | Latest run's output: 20 posts, JSON array of strings. |
| `requirements.txt` | Third-party dependencies. |

## Known limitations

- Kazakh-language output has not been reviewed by a native speaker; treat it as unverified
  before publishing.
- Stage 3's AI scoring is deliberately lenient (a mundane-but-believable post is meant to pass);
  most of the real filtering for "sounds like AI" happens in Stage 4's selection weighting, not
  as a hard rejection.
- The safety regexes are tuned against the current corpus and test cases, not exhaustive —
  review output before publishing, the same way you would review any AI-generated content going
  to real users.
