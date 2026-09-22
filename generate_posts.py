"""UniTee post generator.

Pipeline (diversity is created during generation, then protected during selection):

    style / rules / strategy / real-post corpus
        -> Stage 1  seeds        neutral "what happened / what was noticed" facts, one call per
                                 topic domain, de-duplicated by meaning (embeddings)
        -> Stage 2  generators   14 post types, each with its own objective and shape, small
                                 batches, each post assigned its own seed, language and writer
                                 gender, with real corpus posts as typing-style references
        -> local safety / cleanup / copy detection against the real corpus
        -> Stage 3  AI scoring   same rubric as before, batched, plus a "formulaic" label and a
                                 topic label so no extra classification call is needed
        -> Stage 4  selection    quality + coverage (topic, type, language, length, opener,
                                 ending, shape) - redundancy - concentration, no quotas
        -> validation            hard errors for safety/duplicates, warnings for concentration
        -> generated_posts.json  (JSON array of strings, exactly FINAL_COUNT items)

Run:  python generate_posts.py        (needs OPENAI_API_KEY in the environment)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Optional, Sequence

import openai
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STYLE_FILE = BASE_DIR / "unitee_style_profile.json"
RULES_FILE = BASE_DIR / "generation_rules.json"
STRATEGY_FILE = BASE_DIR / "content_strategy.json"

# Real Telegram posts (read-only). The first file supplies typing-style exemplars; all of them
# are used for copy detection so nothing generated can echo a source post.
CORPUS_FILE = BASE_DIR / "safe_style_posts.json"
EXTRA_CORPUS_FILES = [
    BASE_DIR / "cleaned_telegram_posts.json",
    BASE_DIR / "flagged_telegram_posts.json",
    BASE_DIR / "excluded_posts.json",
    BASE_DIR / "review_posts.json",
]

OUTPUT_FILE = BASE_DIR / "generated_posts.json"
DEBUG_FILE = BASE_DIR / "generation_debug.json"  # candidate pool + reasons, for diagnosing runs

MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512

FINAL_COUNT = 20
QUALITY_THRESHOLD = 5  # unchanged from the previous Stage 3
MAX_ROUNDS = 2  # round 2 is an explicit recovery round

# Seeds per domain by kind. The kind decides which post types may use the seed (TYPE_KINDS):
#   incident = something specific that happened to a student and is worth telling
#   topic    = a general practice / decision / norm students ask about or argue over
#   oddity   = a small odd, funny or contradictory thing someone noticed
#   interest = an activity, show, game or need someone might look for company or help with
SEED_MIX = {"incident": 5, "topic": 5, "oddity": 3, "interest": 3}
GEN_BATCH_SIZE = 5
SCORE_BATCH_SIZE = 20
EXEMPLARS_PER_CALL = 4

MAX_WORKERS = 6
MIN_POST_CHARS = 10

# Similarity thresholds differ by backend (embedding cosine vs. lexical Jaccard).
SIM_THRESHOLDS = {
    "embedding": {"soft": 0.45, "hard": 0.70, "seed": 0.60},
    "lexical": {"soft": 0.30, "hard": 0.55, "seed": 0.45},
}

# Number of candidate posts to request per type in a full round. Roughly follows the shape of
# the real corpus (lots of short questions and one-liners, fewer stories) without being a quota
# on the final output: Stage 4 chooses freely from whatever survives.
TYPE_SLOTS = {
    "quick_ask": 12,
    "info_question": 22,
    "this_or_that": 12,
    "weird_question": 14,
    "find_people": 12,
    "shower_thought": 16,
    "observation": 12,
    "confession": 12,
    "story": 13,
    "awkward_incident": 9,
    "rant": 16,
    "hot_take": 16,
    "relationship_vent": 14,
    "social_awkward": 9,
    "ramble": 5,
}
# Which seed kinds each post type may be built on. Without this every type collapsed into
# "retell the seed event", whatever the type was supposed to be.
TYPE_KINDS = {
    "quick_ask": {"topic", "interest"},
    "info_question": {"topic"},
    "this_or_that": {"topic", "oddity"},
    "weird_question": {"oddity", "topic"},
    "find_people": {"interest"},
    "shower_thought": {"oddity"},
    "observation": {"oddity"},
    "confession": {"incident", "oddity"},
    "story": {"incident"},
    "awkward_incident": {"incident"},
    "rant": {"topic", "incident"},
    "hot_take": {"topic"},
    "relationship_vent": {"incident"},
    "social_awkward": {"incident"},
    "ramble": {"incident", "topic"},
}

# Stage 4 weights (see objective()).
W_QUALITY = 2.0
W_LEN_PRIOR = 1.0  # soft pull toward the length mix of real posts; coverage still spreads it
COVERAGE_WEIGHTS = {  # sqrt-count reward: new values pay well, repeats pay less and less
    "type": 1.0,
    "topic": 1.5,
    "lang": 0.6,
    "lenband": 1.0,
    "opener": 0.8,
    "ending": 0.5,
    "shape": 0.5,
}
# Soft target counts (not hard quotas): concave penalty for drifting from the real corpus's
# proportion, derived at runtime in main() and passed into SelectionContext. A coverage-style
# reward (sqrt of a raw count) rewards HAVING more of a value; that is wrong for a binary habit
# like lowercase-start, where the goal is matching a known real-world rate, not maximizing spread.
W_TARGET = {"casing": 1.4, "emoji": 0.6}
W_SIM = 6.0
W_QUESTION_EXCESS = 1.5
W_ADVICE_EXCESS = 3.0
W_PATTERN_REPEAT = 1.5
W_TELL = 0.35
W_FORMULAIC = 1.2
W_AI_FEEL = 1.5
HARD_CAPS = {"topic": 5, "type": 4, "advice": 4, "opener1": 4, "opener2": 2}
LENGTH_BANDS = [30, 60, 100, 150, 220, 330]  # upper bounds; last band is open-ended


class PipelineError(RuntimeError):
    """Raised for any condition where we prefer failing over publishing weak content."""


class FatalAPIError(PipelineError):
    """API problem that retrying cannot fix (bad key, no quota, unknown model)."""


# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path: Path, *, required: bool = True):
    if not path.exists():
        if required:
            raise PipelineError(f"Missing required file: {path.name}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise PipelineError(f"{path.name} is not valid JSON: {error}") from error


_client: Optional[OpenAI] = None
_client_lock = threading.Lock()


def get_client() -> OpenAI:
    """Lazy client so a missing key produces a clear message instead of an import-time crash."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise FatalAPIError("OPENAI_API_KEY is not set in the environment.")
            _client = OpenAI(api_key=api_key, max_retries=4, timeout=180)
        return _client


def _fatal_or_retryable(error: Exception, purpose: str) -> None:
    """Re-raise errors that retrying cannot fix as FatalAPIError."""
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        raise FatalAPIError(f"OpenAI rejected the credentials ({purpose}): {error}") from error
    if isinstance(error, openai.NotFoundError):
        raise FatalAPIError(f"OpenAI model/endpoint not found ({purpose}): {error}") from error
    if isinstance(error, openai.RateLimitError) and getattr(error, "code", None) == "insufficient_quota":
        raise FatalAPIError(f"OpenAI quota exhausted ({purpose}): {error}") from error
    if isinstance(error, openai.BadRequestError):
        # Per-job failure (counted by run_parallel); if every call fails the run aborts anyway.
        raise PipelineError(f"OpenAI rejected the request ({purpose}): {error}") from error


def call_json(system_prompt: str, user_prompt: str, *, purpose: str,
              max_tokens: int = 6000, attempts: int = 3) -> dict:
    """One JSON-mode chat call with retries for malformed / truncated / empty output.

    Transient API errors (429, 5xx, timeouts) are already retried with backoff inside the SDK;
    the loop here covers what the SDK cannot: bad model output.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            response = get_client().chat.completions.create(
                model=MODEL,
                reasoning_effort="low",
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt + "\n\nYour response MUST be valid JSON."},
                    {"role": "user", "content": user_prompt + "\n\nReturn your answer as valid JSON."},
                ],
            )
            choice = response.choices[0]
            content = (choice.message.content or "").strip()
            if choice.finish_reason == "length":
                raise ValueError("response truncated (finish_reason=length)")
            if not content:
                raise ValueError("empty response")
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON value is not an object")
            return data
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
        except openai.OpenAIError as error:
            _fatal_or_retryable(error, purpose)
            last_error = error
        time.sleep(2 * attempt)
    raise PipelineError(f"{purpose}: giving up after {attempts} attempts ({last_error})")


_print_lock = threading.Lock()


def run_parallel(label: str, func: Callable, jobs: Sequence, *, max_fail_ratio: float = 0.5) -> list:
    """Run jobs concurrently; failed jobs yield None. Fails loudly if most jobs fail."""
    results: list = [None] * len(jobs)
    failures = 0
    done = 0
    if not jobs:
        return results
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = [executor.submit(func, job) for job in jobs]
    try:
        for index, future in enumerate(futures):
            try:
                results[index] = future.result()
            except FatalAPIError:
                raise
            except Exception as error:  # noqa: BLE001 - reported below, never silent
                failures += 1
                with _print_lock:
                    print(f"  [{label}] job {index + 1}/{len(jobs)} failed: {error}", flush=True)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                with _print_lock:
                    print(f"  [{label}] {done}/{len(jobs)} calls finished", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if failures / len(jobs) > max_fail_ratio:
        raise PipelineError(f"{label}: {failures}/{len(jobs)} model calls failed")
    return results


def normalize(text) -> str:
    text = str(text).replace("\u200b", "").strip()
    return re.sub(r"\s+", " ", text)


WORD_RE = re.compile(r"[a-zа-яёәіңғқөұүһ0-9]+", re.IGNORECASE)
STOPWORDS = {
    "и", "в", "на", "не", "что", "я", "с", "а", "но", "как", "это", "то", "у", "мне", "ли", "бы",
    "по", "же", "для", "из", "за", "от", "до", "или", "ну", "все", "так", "да", "нет", "его", "её",
    "the", "a", "to", "of", "is", "and", "it", "i", "in", "my", "me", "you", "for", "on", "at",
    "every", "there", "would", "about", "which", "other", "their", "these", "those", "could",
    "still", "after", "before", "being", "while", "where", "again", "always", "never", "really",
    "мен", "бар", "ма", "ме", "ба", "бе", "па", "пе", "да", "де", "бір", "және",
}


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def stems(text: str) -> set[str]:
    """Crude 5-letter prefixes: enough to match Russian/Kazakh inflections without a stemmer."""
    return {w[:5] for w in words(text) if w not in STOPWORDS and len(w) > 1}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_key(text: str) -> str:
    return re.sub(r"[^a-zа-яәіңғқөұүһё0-9]", "", text.lower())


# ============================================================
# LANGUAGE DETECTION (deterministic, Russian / Kazakh / English aware)
# ============================================================

KZ_LETTERS = set("әіңғқөұүһ")
KZ_WORDS = {
    "бар", "жоқ", "қалай", "неге", "керек", "болды", "мен", "сен", "бүгін", "маған", "үшін",
    "деген", "барма", "ма", "ме", "ба", "бе", "па", "пе", "ғой", "өте", "жақсы", "қазір",
    "кім", "қандай", "қай", "бәрі", "көп", "тағы", "ұнайды", "болады", "жасап", "алмаймын",
    "ешкім", "біреу", "ыстиды", "кормедм", "бугын", "кашан", "нешеге", "кандай", "калай",
    "сондай", "осы", "мына", "ол", "біз", "сіз", "олар", "туралы", "болса", "болмайды",
}


def detect_language(text: str) -> str:
    """Returns one of: ru, kz, ru_kz, ru_en, en."""
    ws = words(text)
    latin = [w for w in ws if re.fullmatch(r"[a-z]+", w) and len(w) >= 2]
    cyr = [w for w in ws if re.search(r"[а-яёәіңғқөұүһ]", w)]
    if not cyr:
        return "en" if latin else "ru"
    if latin:
        return "ru_en"
    kz_hits = sum(1 for w in cyr if (set(w) & KZ_LETTERS) or w in KZ_WORDS)
    fraction = kz_hits / len(cyr)
    if fraction >= 0.45:
        return "kz"
    if fraction >= 0.15 or kz_hits >= 2:
        return "ru_kz"
    return "ru"


# ============================================================
# LOCAL SAFETY / CLEANUP
# ============================================================

def _rx(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
    return re.compile(pattern, flags)


_SOCIAL = (r"(?:инст(?:а|у|е|ой|ы|аграм\w*)?|insta(?:gram)?|тг|tg|телеграм\w*|telegram|whatsapp|"
           r"ватсап\w*|вотсап\w*|снап(?:чат\w*)?|snap(?:chat)?|юз(?:ер\w*|а|у)?|username)")
_PLATFORM = r"(?:" + _SOCIAL + r"|номер\w*|телефон\w*|контакт\w*|number)"

# Multilingual note: Python's \b and \w are Unicode-aware for str patterns, so they work on
# Cyrillic and Kazakh letters. Topical mentions ("телефон сел", "сидел в инсте") are fine; we
# only block contact *exchange*, which is what the old bare-word bans were really after.
SAFETY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("url", _rx(r"https?://|www\.|\b[\w-]+\.(?:com|kz|ru|org|net|me|io|app|gg)\b")),
    ("handle", _rx(r"(?<!\w)@\w{2,}|\b[\w.+-]+@[\w-]+\.\w{2,}\b")),
    ("tme_link", _rx(r"\bt\.me\b")),
    ("contact_request", _rx(
        r"\b(?:кинь|кинуть|кидай|скинь|скиньте|скинуть|дай|дайте|напиши|пиши|пишите|оставь|оставьте|"
        r"send|drop|share|give|find)\W+(?:\w+\W+){0,3}?" + _PLATFORM + r"\b")),
    ("own_contact", _rx(r"\b(?:мой|моя|мои|my|менің)\W+" + _SOCIAL + r"\b")),
    ("contact_label", _rx(r"\b(?:insta|inst|инста|инст|tg|тг|ig|snap|wa|telegram|телеграм\w*)\s*[:=]")),
    ("threat", _rx(r"\b(?:убью|убьем|убьём|прибью|найду и|заставлю пожалеть|сломаю (?:тебе|ему|ей)|"
                   r"kill you|i'?ll find you|өлтіремін)\b")),
    ("self_harm", _rx(r"суицид|самоуб|убить себя|хочу умереть|покончить с собой|kill myself|"
                      r"suicid|өзімді өлтір")),
    ("sexual", _rx(r"\bсекс\w*|\bsex\b|\bnudes?\b|порно|\bporn|\bголая\b|\bголый\b|изнасил|\brape\b")),
    ("slur", _rx(r"пидор\w*|пидар\w*|\bчурк\w*|\bхач(?:и|ей|ам)?\b|\bжид(?:ы|ов|а|у|ом)?\b|\bнигер(?:ы|ов|а)?\b|\bnigg\w+|faggot")),
    ("harassment", _rx(r"\b(?:идиот|идиоты|идиотов|дебил|дебилы|дебилов|мразь|мрази|тварь|твари|"
                       r"шлюха|шлюхи|шлюх|проститутк\w*|ублюд\w+|урод|уроды|уродов)\b")),
    ("username_like", _rx(r"\b[a-z]{2,}[._]?\d{2,}\b")),
    ("initials", _rx(r"\b[А-ЯЁA-Z]\.\s?[А-ЯЁA-Z]\.(?!\w)", 0)),
    # Fake / unverifiable university facts. Generic words (пара, кампус, общага, экзамен) stay legal.
    ("institution_or_city", _rx(
        r"\b(?:сду|sdu|нусом\w{0,3}|nusom|кбту|kbtu|аиту|aitu|iitu|nazarbayev\w*|назарбаев\w*|алмат\w+|"
        r"almaty|астан\w+|astana|нур-?султан|каскелен\w*|kaskelen|шымкент|караганд\w+)\b")),
    ("room_or_building", _rx(
        r"\b[A-Za-zА-Яа-я]-?\d{3}\b|\b\d\.\d{3}\b|"
        r"\b(?:корпус\w*|блок|аудитори\w+|каб(?:инет\w*)?\.?|этаж\w*|building|room|floor)\s*[№#]?\s*[A-Za-zА-Яа-я]?\d+|"
        r"\b\d+[- ]?(?:й|ый|ий)?\s*(?:этаж\w*|корпус\w*|floor)\b")),
    ("named_teacher", re.compile(
        r"(?i:\b(?:препод\w*|преп|профессор\w*|prof(?:essor)?\.?|доцент\w*|учител\w+|тичер\w*|teacher|"
        r"mr\.?|ms\.?|mrs\.?|ағай|апай|мұғалім\w*|оқытушы)\s+)[А-ЯӘІҢҒҚӨҰҮҺA-Z][а-яәіңғқөұүһёa-z]{2,}")),
    ("price", _rx(
        r"\d[\d\s.,]*\s?(?:тг|тенге|теңге|₸|kzt|тыс(?:яч)?\w*|руб\w*|₽|usd|доллар\w*|баксов)\b|"
        r"[₸$₽]\s?\d|\d\s?[$₸₽]|\b\d+\s?[кk]\b(?![a-zа-яё])")),
    ("person_description", _rx(
        r"\bв\s+(?:красн|зелен|зелён|син|черн|чёрн|бел|сер|жёлт|желт|розов|голуб)\w*\s+"
        r"(?:куртк|худи|футболк|штан|плать|кофт|пальто|рубашк|кроссовк|шапк|юбк)\w*")),
]

PHONE_CANDIDATE = re.compile(r"[+\d][\d\s\-().]{7,}\d")

GENDER_PLACEHOLDER_REMOVE = _rx(
    r"\s?\((?:а|я|ая|ой|ый|ий|ое|ые|ла|ась|лась|ся)\)")
GENDER_PLACEHOLDER_LEFT = [
    _rx(r"[a-zа-яёәіңғқөұүһ]\((?:а|я|ая|ой|ый|ий|ое|ые|ла|ась|лась|ся|ка|ён|ен|ена)\)"),
    _rx(r"\b[а-яё]{2,}/(?:а|я|ла|лась|ась|ая|ый|ой|ий)\b"),
]

# Title-cased words that are fine mid-sentence (brands, languages, calendar words).
ALLOWED_CAPS = {
    "tiktok", "instagram", "netflix", "youtube", "spotify", "telegram", "whatsapp", "google",
    "chatgpt", "gpt", "zoom", "excel", "word", "powerpoint", "python", "kaspi", "yandex",
    "samsung", "iphone", "android", "macbook", "pinterest", "discord", "steam", "twitter",
    "snapchat", "ielts", "toefl", "gpa", "wifi", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december", "english", "russian", "kazakh",
    "korean", "japanese", "chinese", "german", "turkish", "french", "spanish", "ramadan",
    "ramazan", "nowruz", "наурыз", "рамазан", "господи", "бог", "аллах", "netflix", "reels",
    "notion", "canva", "figma", "linkedin", "tinder", "uber", "yandex", "glovo", "wolt",
}
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәІіҢңҒғҚқӨөҰұҮүҺһ][\w'’-]*")


def has_probable_name(text: str) -> bool:
    """A title-cased word in mid-sentence is most likely a person/brand name."""
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) < 3 or not token[0].isupper() or token.isupper():
            continue
        if token[1:] != token[1:].lower():
            continue
        if token.lower() in ALLOWED_CAPS:
            continue
        before = text[: match.start()]
        if re.search(r"(?:^|[.!?…]\s+|[\"«“(]\s*)$", before):
            continue  # sentence start
        return True
    return False


# Distinctive Kazakh/Russian/Turkic given names (prefix-matched, so inflected forms are caught).
# Capitalization cannot catch a name at the start of a sentence or in all-lowercase typing.
# Names that are also common words (Вера, Надежда, Роза, Мира...) are deliberately left out.
GIVEN_NAMES = (
    "айгерим айгуль айдана айбек айдар айжан айнур айсултан айсулу акбота акерке алия алишер алмас "
    "алина альбина амина амир анара аниса арман арсен арайлан асель асем асхат аян ажар бауыржан "
    "бекжан берик даниал данияр даурен динара дана дарын дильназ досжан думан еркебулан ернар "
    "жандос жанар жания заур зере зульфия камила канат карина кайрат кристина ляззат мадина мадияр "
    "малика мейирим мейіржан мурат мухаммед нургуль нурбол нурбакыт нурислам нурсултан нуркен "
    "руслан рахат самат санжар сания сабина серик султан тимур томирис улжан улпан фариза шарип "
    "шариф эльмира диана дамир дархан алихан адиль азамат азат алмат арсений богдан вадим денис "
    "дмитрий евгений ильдар ильяс камшат ксения лейла марат милана назерке ханзада хадиша ярослав "
    "daniyar aigerim aidana aibek aidar asel aselya alisher arman arsen madina madiyar dana darin "
    "dauren dinara dilnaz erlan nurislam nursultan ruslan samat sanzhar temirlan timur zhanar "
    "kanat kairat kamila karina amir amina anara alinur azamat azat almas"
).split()
GIVEN_NAME_RE = re.compile(r"\b(?:" + "|".join(sorted(set(GIVEN_NAMES), key=len, reverse=True)) + r")\w{0,3}\b",
                           re.IGNORECASE)


def has_phone_number(text: str) -> bool:
    for match in PHONE_CANDIDATE.finditer(text):
        if sum(ch.isdigit() for ch in match.group(0)) >= 9:
            return True
    return False


def find_violation(text: str, *, check_names: bool = True) -> Optional[str]:
    """Return the name of the first safety rule the text breaks, or None."""
    if "—" in text or "–" in text:
        return "dash"
    if has_phone_number(text):
        return "phone"
    for name, pattern in SAFETY_PATTERNS:
        if pattern.search(text):
            return name
    if check_names:
        if any(p.search(text) for p in GENDER_PLACEHOLDER_LEFT):
            return "gender_placeholder"
        if has_probable_name(text) or GIVEN_NAME_RE.search(text):
            return "possible_person_name"
    return None


def clean_post_text(raw) -> str:
    """Whitespace, wrapping quotes, list numbering, and (а)-style placeholders."""
    if not isinstance(raw, str):
        return ""
    text = normalize(raw)
    text = re.sub(r"\s*[—–]\s*", " - ", text)  # real typed posts use a spaced hyphen; em/en dashes are banned
    text = re.sub(r"^\s*\d{1,3}[.)]\s+", "", text)
    if len(text) > 2 and text[0] in "\"«“" and text[-1] in "\"»”":
        text = text[1:-1].strip()
    text = GENDER_PLACEHOLDER_REMOVE.sub("", text)
    return normalize(text)


# ============================================================
# COPY DETECTION AGAINST THE REAL CORPUS
# ============================================================

class SourceIndex:
    """Flags generated text that reuses a run of source wording or is a lexical near-copy."""

    def __init__(self, texts: Sequence[str]):
        self.entries = []
        for text in texts:
            ws = words(text)
            self.entries.append((stems(text), {" ".join(ws[i:i + 5]) for i in range(len(ws) - 4)}))

    def copy_reason(self, text: str) -> Optional[str]:
        ws = words(text)
        grams = {" ".join(ws[i:i + 5]) for i in range(len(ws) - 4)}
        cand_stems = stems(text)
        for source_stems, source_grams in self.entries:
            if grams & source_grams:
                return "copies_source"
            if len(cand_stems) >= 4 and jaccard(cand_stems, source_stems) >= 0.55:
                return "copies_source"
        return None


# ============================================================
# CORPUS / EXEMPLARS
# ============================================================

DIRECT_ADDRESS = _rx(r"\b(?:ты|тебя|тебе|тобой|сен|сені|саған|you|ur)\b")
HEAVY_PROFANITY = _rx(r"хуй|хуе|ебан|еблан|ебат|пизд|\bбля|сука|fuck|shit|bitch")
SERIES_POST = _rx(r"\btop ?\d|\bтоп ?\d")
# Real posts are full of local specifics (named clubs, events, ID-card hunts, university slang) and
# some sexual / orientation content. Showing those to the model as "typing references" invites
# invented local facts, so they are kept out of the prompt exemplars (they still feed copy detection).
EXEMPLAR_EXCLUDE = _rx(
    r"клуб|клаб|club|\bуник|универ|сду\b|нусом|бхб|\bдх\b|айди|рус пед|дейлик|мюзик|кбту|аегис|"
    r"яндекс тим|перваш|@|kiss|\bgay|яой|\bx\b|кастинг|коттедж|\bavatar|\bаву\b|\bавы\b|яойк|"
    r"ltn|blasian|тютор|\bдорам|\bкорды|убро|\bдоры|\bдшд|фараонд|\bикт\b|\bwt\b")


def load_corpus() -> tuple[list[dict], SourceIndex]:
    data = load_json(CORPUS_FILE)
    posts = data.get("posts") if isinstance(data, dict) else None
    if not posts:
        raise PipelineError(f"{CORPUS_FILE.name} has no posts.")
    all_texts = [normalize(p.get("text", "")) for p in posts if isinstance(p, dict)]
    for path in EXTRA_CORPUS_FILES:
        extra = load_json(path, required=False)
        if isinstance(extra, dict):
            all_texts += [normalize(p.get("text", "")) for p in extra.get("posts", []) if isinstance(p, dict)]
    return posts, SourceIndex([t for t in all_texts if t])


class ExemplarPool:
    """Real posts that are safe to show the model as typing-style references.

    Rotates through the pool (least-used first) so no single source post dominates the prompts,
    which keeps both copying and topic leakage down.
    """

    def __init__(self, posts: list[dict], flagged_texts: set[str]):
        self.items: list[str] = []
        self.lang: dict[str, str] = {}
        for post in posts:
            text = normalize(post.get("text", ""))
            if not (15 <= len(text) <= 420) or text in flagged_texts:
                continue
            if find_violation(text) or DIRECT_ADDRESS.search(text) or HEAVY_PROFANITY.search(text):
                continue
            if SERIES_POST.search(text) or EXEMPLAR_EXCLUDE.search(text):
                continue
            self.items.append(text)
            self.lang[text] = detect_language(text)
        self.used: Counter = Counter()

    def sample(self, hint: tuple[int, int, Optional[bool]], language: str, k: int,
               rng: random.Random) -> list[str]:
        """Prefer exemplars in the batch's language, then ones matching the type's length hint."""
        low, high, question = hint
        same_lang = [t for t in self.items if self.lang[t] == language]
        if len(same_lang) < 2:
            same_lang = [t for t in self.items if self.lang[t] in ("ru", "ru_kz")] if language != "en" else self.items
        fits = [t for t in same_lang
                if low <= len(t) <= high and (question is None or ("?" in t) == question)]
        matching = fits if len(fits) >= min(k, 2) else same_lang
        matching = matching[:]
        rng.shuffle(matching)
        matching.sort(key=lambda t: self.used[t])
        chosen = matching[:k]
        for text in chosen:
            self.used[text] += 1
        return chosen


# ============================================================
# HARD LIMITS TEXT (built from generation_rules.json so config edits take effect)
# ============================================================

def build_hard_limits(rules: dict) -> str:
    try:
        never_invent = ", ".join(rules["factual_information"]["never_invent"])
        never_generate = ", ".join(rules["safety"]["never_generate"])
    except KeyError as error:
        raise PipelineError(f"generation_rules.json is missing expected key: {error}") from error
    return (
        f"- Never state invented facts about a university: {never_invent}. Never name a person, "
        "teacher, club, building, room, cafe, event, city or university. Refer to people "
        "generically (a friend, my roommate, a classmate, this guy).\n"
        f"- Never write: {never_generate}. No links, handles, phone numbers or contact requests.\n"
        "- No em dashes or en dashes.\n"
        "- The writer's gender is given per post. Use matching verb/adjective forms, or phrase "
        "around gender when it is unspecified. Never write forms like устал(а) or рад/а."
    )


# ============================================================
# STAGE 1: SEEDS
# ============================================================

DOMAINS = {
    "exams_grades": "exams, deadlines, grades, retakes, attendance, homework, cramming (generic: no named courses, teachers or policies)",
    "classes_teachers": "what happens in lectures and seminars, groupmates, group projects, generic teacher behavior, note-taking",
    "dorm_home_life": "roommates, shared spaces, noise, laundry, cooking, living away from home, commuting",
    "food": "eating habits, cooking, delivery, snacks, drinks, cravings, cheap meals",
    "looks_style": "clothes, skincare, hair, gym body, shopping, self-image, fashion opinions",
    "health_routine": "sleep, gym, sports, sickness, daily routine, energy, habits",
    "romance": "crushes, talking stages, dating, exes, jealousy, texting habits (never an identifiable person)",
    "friends_social": "friendships, groups, plans, awkward encounters, favors, lending things, group chats",
    "family_home": "parents, relatives, hometown, expectations, homesickness, money from home",
    "money_work": "part-time work, spending, saving, tutoring, freelancing, scholarships in general terms",
    "entertainment": "anime, dorama, music, games, books, movies, series, fandoms, short videos",
    "internet_tech": "phones, apps, wifi, AI tools for homework, social media habits, gadgets",
    "identity_culture": "language and code-switching, stereotypes, regional differences, generation gaps, traditions, opinions on social norms",
    "future_abroad": "career plans, internships, exchange programs, language exams, moving abroad, choosing a major",
    "clubs_activities": "joining activities in general, volunteering, events in general, hobbies, meeting new people",
    "random_life": "odd thoughts, coincidences, transport, weather, city life, small everyday mysteries",
}

# Angles rotated into each seed call so different calls explore different kinds of moments.
LENSES = [
    "something someone noticed about other people",
    "a small mistake or mix-up",
    "a plan that fell through",
    "a purchase or something that broke",
    "a conversation that went sideways",
    "a habit the person only recently became aware of",
    "a rule or norm they quietly disagree with",
    "a coincidence",
    "something they overheard or saw online",
    "a comparison between two options",
    "a thing that is oddly specific and mundane",
    "something they are avoiding doing",
    "a tiny win",
    "something that changed since first year",
]


@dataclass
class Seed:
    id: int
    domain: str
    kind: str          # incident | topic | oddity
    theme: str
    subject: str
    detail: str
    embedding: Optional[list[float]] = None


KIND_DESCRIPTIONS = {
    "incident": "something specific that happened to a student and is worth telling because it was "
                "funny, annoying, awkward, surprising or oddly specific (NOT a bland routine change "
                "like 'started cooking' or 'stopped drinking soda'). One sentence for what happened, "
                "plus two concrete details (what led up to it, what someone said, an object, a "
                "number, what happened right after) so there is enough material for a writer who "
                "wants to tell it at length, even though most posts will still use only part of it.",
    "topic": "a general practice, decision, norm or debate students ask each other about or argue "
             "over. A SHORT NOUN PHRASE (max 10 words), no detail, no single event.",
    "oddity": "a small odd, funny, absurd or contradictory thing people do or say. A SHORT NOUN "
              "PHRASE (max 10 words), no detail.",
    "interest": "an activity, hobby, game, show, exam prep or need someone might look for company "
                "or help with. A SHORT NOUN PHRASE (max 8 words), no detail. If the domain has "
                "none, use something adjacent.",
}


def generate_seeds_for_domain(job: tuple, hard_limits: str) -> list[dict]:
    domain, lenses, avoid_themes = job
    system_prompt = (
        "You collect raw material for anonymous posts by university students in Kazakhstan "
        "(Russian / Kazakh / English speaking). You produce SEEDS: neutral notes about one "
        "specific subject. The seeds are background material for a writer, not posts.\n\n"
        "A seed states the subject only: no feelings, no question to the reader, no advice-seeking, "
        "no moral, no invented reaction. Every seed must be a different underlying situation. If "
        "two seeds would lead to the same post ('friend ignores me' and 'friend became distant' "
        "are ONE situation), keep only one. Every seed needs a hook: it is surprising, annoying, "
        "funny, awkward, contested or oddly specific. Skip bland self-improvement and routine.\n"
        "Think of what students actually post: a skincare product, how an exam is graded, a crush's "
        "texting, cheap food, sleep, gym, a show's ending, an awkward message, money for lunch, a "
        "stereotype. Small, dumb, personal. NOT projects, experiments, initiatives, organizing, "
        "events, policies or ways to improve student life.\n\n"
        "Hard limits:\n" + hard_limits
    )
    avoid = ""
    if avoid_themes:
        avoid = "\nThemes already covered elsewhere, do not repeat: " + "; ".join(avoid_themes[:60]) + "\n"
    kinds = "\n".join(f"- {kind} ({count} seeds): {KIND_DESCRIPTIONS[kind]}"
                       for kind, count in SEED_MIX.items())
    user_prompt = (
        f"DOMAIN: {domain}\nScope: {DOMAINS[domain]}\n\n"
        f"Write exactly {sum(SEED_MIX.values())} seeds in these kinds:\n{kinds}\n\n"
        "Cover a range of moments, for example:\n- " + "\n- ".join(lenses) + "\n" + avoid +
        "\nReturn JSON: {\"seeds\": [{\"kind\": \"incident|topic|oddity|interest\", "
        "\"theme\": \"2-5 word label for the kind of situation\", "
        "\"subject\": \"English; a sentence for incidents, a short noun phrase for the other kinds\", "
        "\"detail\": \"two concrete details separated by a semicolon, English; ONLY for incidents, otherwise an empty string\"}]}"
    )
    data = call_json(system_prompt, user_prompt, purpose=f"seeds/{domain}", max_tokens=6000)
    items = data.get("seeds")
    if not isinstance(items, list):
        raise PipelineError(f"seeds/{domain}: response has no 'seeds' list")
    return [{**item, "domain": domain} for item in items if isinstance(item, dict)]


def stage1_seeds(rng: random.Random, hard_limits: str, *, start_id: int,
                 avoid_themes: list[str], stats: dict) -> list[Seed]:
    jobs = [(domain, rng.sample(LENSES, 5), avoid_themes) for domain in DOMAINS]
    print(f"\nStage 1: requesting {sum(SEED_MIX.values())} seeds x {len(DOMAINS)} domains...", flush=True)
    results = run_parallel("seeds", lambda job: generate_seeds_for_domain(job, hard_limits), jobs)

    seeds: list[Seed] = []
    rejected = Counter()
    next_id = start_id
    for batch in results:
        for item in batch or []:
            kind = normalize(item.get("kind", "")).lower()
            theme, subject, detail = (normalize(item.get(k, "")) for k in ("theme", "subject", "detail"))
            if kind not in KIND_DESCRIPTIONS or not theme or not subject:
                rejected["malformed"] += 1
                continue
            if find_violation(f"{subject} {detail}", check_names=False):
                rejected["unsafe_or_fake_fact"] += 1
                continue
            seeds.append(Seed(next_id, item["domain"], kind, theme, subject,
                              detail if kind == "incident" else ""))
            next_id += 1
    stats["seeds_generated"] = stats.get("seeds_generated", 0) + len(seeds)
    print(f"  seeds returned: {len(seeds)} (rejected: {dict(rejected) or 0})")
    return seeds


# ============================================================
# EMBEDDINGS (optional; lexical fallback keeps the pipeline running)
# ============================================================

def embed_texts(texts: Sequence[str]) -> Optional[list[list[float]]]:
    """Unit-length embeddings, or None if the embedding call fails (caller falls back)."""
    if not texts:
        return []
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), 200):
            response = get_client().embeddings.create(
                model=EMBED_MODEL, input=list(texts[start:start + 200]), dimensions=EMBED_DIMS)
            for item in response.data:
                norm = math.sqrt(sum(x * x for x in item.embedding)) or 1.0
                vectors.append([x / norm for x in item.embedding])
    except FatalAPIError:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"  WARNING: embeddings unavailable ({error}); falling back to lexical similarity.")
        return None
    return vectors


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def dedupe_seeds(seeds: list[Seed], stats: dict) -> tuple[list[Seed], str]:
    """Keep a seed only if its meaning differs from every already-kept seed.

    Seeds are processed round-robin across domains so every domain gets a fair share.
    """
    backend = "embedding"
    vectors = embed_texts([f"{s.theme}. {s.subject}" for s in seeds])
    if vectors is None:
        backend = "lexical"
    else:
        for seed, vector in zip(seeds, vectors):
            seed.embedding = vector
    threshold = SIM_THRESHOLDS[backend]["seed"]

    by_domain: dict[str, list[Seed]] = {}
    for seed in seeds:
        by_domain.setdefault(seed.domain, []).append(seed)
    ordered = [s for group in zip_longest(*by_domain.values()) for s in group if s is not None]

    kept: list[Seed] = []
    for seed in ordered:
        if backend == "embedding":
            too_close = any(dot(seed.embedding, k.embedding) >= threshold for k in kept)
        else:
            mine = stems(f"{seed.theme} {seed.subject}")
            too_close = any(jaccard(mine, stems(f"{k.theme} {k.subject}")) >= threshold for k in kept)
        if not too_close:
            kept.append(seed)
    stats["seeds_unique"] = stats.get("seeds_unique", 0) + len(kept)
    return kept, backend


# ============================================================
# STAGE 2: POST GENERATORS
# ============================================================

@dataclass
class PostType:
    name: str
    objective: str          # what the person is doing by posting
    shape: str              # structural tendency (never a character target)
    avoid: str              # what this type specifically does not do
    affinity: set[str]      # domains whose seeds suit this type (empty = any)
    exemplar_hint: tuple[int, int, Optional[bool]]  # (min_len, max_len, question?) for exemplars


POST_TYPES: dict[str, PostType] = {t.name: t for t in [
    PostType(
        "quick_ask",
        "fires off a fragment of a question or request with no context, the way one types into a "
        "group chat while doing something else.",
        "two to seven words, a fragment; no greeting, no context, may lack a verb or a question mark.",
        "no explanation, no politeness, no second sentence.",
        {"exams_grades", "classes_teachers", "looks_style", "money_work", "future_abroad", "food",
         "health_routine", "entertainment", "internet_tech", "dorm_home_life", "clubs_activities"},
        (15, 45, None)),
    PostType(
        "info_question",
        "wants a concrete answer from other students: how something works, what to pick, whether "
        "something is worth it, what people usually do about X.",
        "ONE sentence, occasionally two: the question itself with at most a few words of context. "
        "Neutral and practical, reads like a search query typed to people.",
        "no emotional lead-in, no 'I feel', no relationship or friendship dilemmas, no life advice.",
        {"exams_grades", "classes_teachers", "looks_style", "money_work", "future_abroad",
         "internet_tech", "food", "health_routine", "clubs_activities", "dorm_home_life", "entertainment"},
        (15, 110, True)),
    PostType(
        "this_or_that",
        "starts a low-stakes argument or poll: pick one, rank things, settle a preference.",
        "a single line with two options or one blunt either/or. No backstory.",
        "no feelings, no explanation of why they are asking.",
        {"food", "looks_style", "entertainment", "dorm_home_life", "health_routine", "internet_tech",
         "identity_culture", "romance", "random_life", "friends_social"},
        (15, 90, True)),
    PostType(
        "weird_question",
        "asks something odd or a bit dumb that popped into their head and actually wants to know.",
        "one or two sentences, oddly literal and specific (bodies, habits, objects, dreams, "
        "language, social norms).",
        "not philosophical, not advice-seeking, no 'why do people...' about people in general.",
        {"random_life", "health_routine", "internet_tech", "food", "identity_culture", "entertainment", "looks_style"},
        (15, 100, True)),
    PostType(
        "find_people",
        "is looking for people: someone to do something with, someone who shares a niche interest, "
        "or someone to talk to.",
        "one or two sentences that open or center on an explicit call to join (кто хочет / ищу "
        "того кто / anyone up for / looking for someone who). If the subject cannot support a real "
        "call to join, skip it - this type is not a disguised info_question.",
        "no soul-searching about loneliness, no names, no contact details.",
        {"entertainment", "clubs_activities", "health_routine", "food", "dorm_home_life",
         "friends_social", "future_abroad"},
        (15, 90, None)),
    PostType(
        "shower_thought",
        "a passing thought, joke or 'me when' moment typed in five seconds.",
        "a single short sentence or fragment (a dozen words is typical). No setup, no explanation, "
        "no question. Can be dumb, or funny by accident. May lack punctuation.",
        "not a proverb, not a lesson, not a question aimed at the audience.",
        set(),
        (15, 70, False)),
    PostType(
        "observation",
        "says something odd or funny they noticed about people, habits, language or the internet.",
        "one or two sentences, first hand and plain. It can end right there, unresolved; the "
        "oddness is the whole point.",
        "no third-person reporting ('a student did X'), never turns it into 'why is this / is it "
        "just me / what should I do', no moral.",
        set(),
        (25, 160, False)),
    PostType(
        "confession",
        "admits something petty, embarrassing, irrational or guilty, just to say it anonymously.",
        "one to three sentences: a first-person admission, blunt, sometimes with a short reason or a "
        "telling detail; may be funny or shameful.",
        "no request for advice or reassurance, no 'is this normal', no growth reflection.",
        set(),
        (20, 180, None)),
    PostType(
        "story",
        "tells a small thing that actually happened, in order.",
        "four to eight plain sentences telling it in order: the lead-up, what happened, what "
        "happened right after. Use the concrete details given (objects, roughly when, unnamed "
        "people, what someone said). Real incidents worth telling usually take some room, commonly "
        "250-500 characters; do not compress a multi-step incident into one line, and do not pad a "
        "thin one to get there. The ending is flat or abrupt, never an engineered twist.",
        "no closing lesson, no summary of feelings, no cute twist, no 'what should I do'.",
        set(),
        (150, 420, None)),
    PostType(
        "awkward_incident",
        "a short embarrassing or funny moment that just happened.",
        "one to three sentences: the awkward thing itself. The cringe lives in the details, not in "
        "stated feelings.",
        "no 'I felt embarrassed and confused', no advice request.",
        {"classes_teachers", "friends_social", "dorm_home_life", "random_life", "food", "romance",
         "internet_tech", "looks_style", "health_routine"},
        (40, 200, None)),
    PostType(
        "rant",
        "complains because something is annoying them right now.",
        "one to three short sentences, blunt and irritated, specific about WHAT is annoying (a "
        "thing, habit, situation, rule). May be repetitive or exaggerated. Mild swearing is fine.",
        "no 'how to solve this', no balanced 'but maybe they have reasons', never attacks an "
        "identifiable person.",
        {"exams_grades", "classes_teachers", "dorm_home_life", "internet_tech", "money_work", "food",
         "random_life", "family_home", "identity_culture"},
        (20, 200, None)),
    PostType(
        "hot_take",
        "states an opinion some people will disagree with, about the subject in general.",
        "one or two sentences, flat and confident, or slightly petty. A claim about how things "
        "are or should be, not a story about the poster's day.",
        "no anecdote, no 'what do you think', no 'am I wrong', no hedging paragraph, no telling "
        "the reader what to do ('please stop doing X', 'respect people') - state the opinion, "
        "don't moralize at the audience.",
        {"identity_culture", "romance", "food", "entertainment", "looks_style", "friends_social",
         "money_work", "internet_tech"},
        (25, 200, False)),
    PostType(
        "relationship_vent",
        "talks about a crush, talking stage, ex or relationship (unnamed, unidentifiable).",
        "one to four sentences saying what is going on or how it feels, plainly: a complaint, a "
        "confession, or a single detail. Only sometimes asks anything, and then something specific "
        "about one behaviour.",
        "no names, no describing a person's looks/clothes/location, no generic 'what should I do', "
        "not every sentence is a feeling.",
        {"romance", "friends_social"},
        (60, 300, None)),
    PostType(
        "social_awkward",
        "friends, groupmates, roommates or strangers: something small happened socially.",
        "two to four sentences: what someone did or said, and what the poster did or did not do. One "
        "sharp specific event; ends without a resolution and without asking what to do.",
        "no vague 'my friend changed lately', no therapy vocabulary.",
        {"friends_social", "classes_teachers", "dorm_home_life", "clubs_activities", "romance"},
        (60, 300, None)),
    PostType(
        "ramble",
        "finally writes down something that has been weighing on them: a situation with a person, a "
        "decision, a family expectation, a feeling that has lasted.",
        "one earnest paragraph, typically five sentences or more and often 300-500 characters: "
        "what the situation is, several specific facts or moments (not just one), and where the "
        "poster is with it now. Plain sincere wording, not performed messiness, but let it take the "
        "room it needs rather than wrapping up in two lines.",
        "no filler like 'idk' or 'ugh', no theatrical stream of consciousness, no neat three-part "
        "arc, no closing question to the audience.",
        {"family_home", "romance", "friends_social", "future_abroad", "identity_culture",
         "exams_grades", "money_work"},
        (200, 420, None)),
]}

LANGUAGE_DESCRIPTIONS = {
    "ru": "Russian, casual typing.",
    "ru_kz": "Russian with some Kazakh words or short phrases dropped in, the way Kazakhstani students actually talk. No English.",
    "kz": "Kazakh (Cyrillic), casual spoken register with the Russian loanwords a student would really use; informal spelling is fine, no bookish Kazakh. No English.",
    "ru_en": "Russian with a few English words mixed in the way students do: slang, fillers and set phrases (like, literally, pls, wtf, tho, deadline). Everything else stays Russian; do not swap ordinary nouns for English ones. No Kazakh.",
    "en": "English, casual, as a fluent non-native student types on a phone. No Russian or Kazakh.",
}
SPEAKER_DESCRIPTIONS = {
    "m": "male (masculine forms)",
    "f": "female (feminine forms)",
    "x": "gender not shown (avoid gendered forms)",
}
DEFAULT_LANGUAGE_WEIGHTS = {"ru": 0.44, "ru_kz": 0.10, "kz": 0.14, "ru_en": 0.14, "en": 0.18}


def derive_language_weights(corpus_posts: list[dict]) -> dict[str, float]:
    """Language mix from the real corpus (the style profile lumps Kazakh into 'cyrillic'),
    clamped so Kazakh - the mode I can check least - never dominates."""
    counts = Counter(detect_language(normalize(p.get("text", ""))) for p in corpus_posts)
    total = sum(counts.values())
    if not total:
        return dict(DEFAULT_LANGUAGE_WEIGHTS)
    weights = {lang: max(0.06, counts.get(lang, 0) / total) for lang in LANGUAGE_DESCRIPTIONS}
    weights["kz"] = min(weights["kz"], 0.16)
    weights["ru"] = max(weights["ru"], 0.36)
    norm = sum(weights.values())
    return {lang: w / norm for lang, w in weights.items()}


@dataclass
class Assignment:
    post_type: str
    seed: Seed
    speaker: str
    lowercase: bool = False  # typing habit sampled at the rate seen in the real corpus
    emoji: bool = False


@dataclass
class Batch:
    """One generation call: one post type, one language, a few seed assignments."""
    post_type: str
    language: str
    items: list[Assignment]


@dataclass
class Candidate:
    id: int
    text: str
    post_type: str
    seed: Seed
    assigned_language: str
    speaker: str
    round: int
    status: str = "generated"
    score: Optional[float] = None
    reason: str = ""
    formulaic: bool = False
    ai_feel: bool = False
    topic: str = ""
    language: str = ""
    features: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None


def plan_batches(seeds: list[Seed], rng: random.Random, language_weights: dict[str, float],
                 scale: float, stats: dict, habit_rates: dict[str, float]) -> list[Batch]:
    """Give every slot its own seed of a suitable kind, then group slots into per-language batches.

    Every seed is used at most once, so two candidates can never be built on the same
    underlying situation. Types with the narrowest seed requirements are served first; a slot
    with no suitable seed left is dropped rather than filled with a mismatched one.
    """
    pool: dict[tuple[str, str], list[Seed]] = {}
    for seed in seeds:
        pool.setdefault((seed.domain, seed.kind), []).append(seed)
    for group in pool.values():
        rng.shuffle(group)

    def take(post_type: PostType) -> Optional[Seed]:
        kinds = TYPE_KINDS[post_type.name]
        options = [key for key, group in pool.items()
                   if group and key[1] in kinds and (not post_type.affinity or key[0] in post_type.affinity)]
        if not options:  # relax domain affinity, never the seed kind
            options = [key for key, group in pool.items() if group and key[1] in kinds]
        if not options:
            return None
        best = max(options, key=lambda key: (len(pool[key]), rng.random()))
        return pool[best].pop()

    slots = {name: max(2, round(count * scale)) for name, count in TYPE_SLOTS.items()}
    order = sorted(slots, key=lambda n: (len(TYPE_KINDS[n]), len(POST_TYPES[n].affinity) or 99, rng.random()))
    by_type: dict[str, list[Assignment]] = {name: [] for name in slots}
    dropped = 0
    while any(slots.values()):  # one slot per type per sweep, so no type drains shared seeds first
        for name in order:
            if slots[name] <= 0:
                continue
            slots[name] -= 1
            seed = take(POST_TYPES[name])
            if seed is None:
                dropped += 1
                continue
            by_type[name].append(Assignment(
                name, seed, rng.choices(["m", "f", "x"], [45, 45, 10])[0],
                lowercase=rng.random() < habit_rates["lowercase"], emoji=rng.random() < habit_rates["emoji"]))
    stats["slots_dropped_no_seed"] = stats.get("slots_dropped_no_seed", 0) + dropped

    # Long free-form Kazakh (a full incident/opinion paragraph) was the weakest output in testing:
    # coherent in short bursts, prone to running on and losing sense at paragraph length. Until a
    # native speaker reviews it, keep Kazakh to the types that stayed short and clean, and let those
    # types carry ru_kz code-switching instead (which read naturally) for the longer ones.
    NO_PURE_KZ = {"story", "ramble", "social_awkward", "relationship_vent"}
    langs, lang_p = zip(*language_weights.items())
    short_langs = [l for l in langs if l != "kz"]
    short_p = [w for l, w in zip(langs, lang_p) if l != "kz"]
    norm = sum(short_p)
    short_p = [w / norm for w in short_p]
    batches = []
    for name, items in by_type.items():
        choices, weights = (short_langs, short_p) if name in NO_PURE_KZ else (langs, lang_p)
        for start in range(0, len(items), GEN_BATCH_SIZE):
            batches.append(Batch(name, rng.choices(choices, weights)[0], items[start:start + GEN_BATCH_SIZE]))
    return batches


GEN_SYSTEM_TEMPLATE = """You write anonymous posts for UniTee, an anonymous app where university students post to a feed of other students, in Russian, Kazakh, English or a mix. Each post is typed once, on a phone, by a different person who has not read any other post. These people are not writers and are not trying to be relatable, deep or engaging.

The poster speaks for themselves and to other students, first hand. Never report a situation in third person like a news summary, never explain background the readers do not need, never tie things up neatly. Write only what the post type and the subject call for. Let each post be as long as its thought is; never pad.

Hard limits:
{hard_limits}"""


def build_generation_prompt(post_type: PostType, batch: Batch, exemplars: list[str]) -> str:
    lines = []
    for number, item in enumerate(batch.items, start=1):
        detail = f" Detail: {item.seed.detail}." if item.seed.detail else ""
        typing = " Typing: all lowercase, little punctuation, no final period." if item.lowercase \
            else " Typing: starts with a capital letter, like most posts do."
        if item.emoji:
            typing += " Includes one emoji, placed the way students do."
        lines.append(f"{number}. [{item.seed.kind}] {item.seed.subject.rstrip('.')}.{detail} "
                     f"Writer: {SPEAKER_DESCRIPTIONS[item.speaker]}.{typing}")
    example_block = "\n".join(f"- {text}" for text in exemplars)
    return (
        f"POST TYPE: {post_type.name}\n"
        f"What this person is doing by posting: {post_type.objective}\n"
        f"How these posts usually look: {post_type.shape}\n"
        f"What this type does NOT do: {post_type.avoid}\n\n"
        f"LANGUAGE for all posts in this batch: {LANGUAGE_DESCRIPTIONS[batch.language]}\n\n"
        "Real posts from the platform. They show typing habits only; never reuse their topics, "
        f"wording, names or jokes:\n{example_block}\n\n"
        f"Write {len(batch.items)} posts, one per subject, each by a different person. The subjects "
        "are notes for you in English: do not translate or retell them, do not carry over their "
        "English wording, and use only as much of the subject as this person would. Invent what "
        "this person specifically says.\n\n"
        + "\n".join(lines) +
        "\n\nReturn JSON: {\"posts\": [{\"n\": 1, \"text\": \"...\"}]}. If a subject cannot "
        "become this kind of post naturally, return {\"n\": 3, \"skip\": true} for it instead of "
        "forcing it."
    )


def generate_batch(job: tuple, system_prompt: str) -> list[dict]:
    batch, exemplars = job
    data = call_json(system_prompt, build_generation_prompt(POST_TYPES[batch.post_type], batch, exemplars),
                     purpose=f"generate/{batch.post_type}", max_tokens=6000)
    items = data.get("posts")
    if not isinstance(items, list):
        raise PipelineError(f"generate/{batch.post_type}: response has no 'posts' list")
    return items


def stage2_generate(batches: list[Batch], exemplars: ExemplarPool, rng: random.Random,
                    hard_limits: str, round_number: int, next_id: int, stats: dict) -> list[Candidate]:
    system_prompt = GEN_SYSTEM_TEMPLATE.format(hard_limits=hard_limits)
    jobs = [(batch, exemplars.sample(POST_TYPES[batch.post_type].exemplar_hint, batch.language,
                                     EXEMPLARS_PER_CALL, rng)) for batch in batches]
    rng.shuffle(jobs)
    total = sum(len(b.items) for b in batches)
    print(f"\nStage 2: {total} assignments across {len({b.post_type for b in batches})} post types "
          f"in {len(jobs)} small single-language calls...", flush=True)
    results = run_parallel("generate", lambda job: generate_batch(job, system_prompt), jobs)

    candidates: list[Candidate] = []
    skipped = malformed = 0
    for (batch, _), items in zip(jobs, results):
        for item in items or []:
            if not isinstance(item, dict) or not isinstance(item.get("n"), int) \
                    or not 1 <= item["n"] <= len(batch.items):
                malformed += 1
                continue
            if item.get("skip") is True:
                skipped += 1
                continue
            text = clean_post_text(item.get("text"))
            if not text:
                malformed += 1
                continue
            assignment = batch.items[item["n"] - 1]
            candidates.append(Candidate(
                next_id + len(candidates), text, batch.post_type, assignment.seed,
                batch.language, assignment.speaker, round_number))
    stats["model_skipped"] = stats.get("model_skipped", 0) + skipped
    stats["malformed_items"] = stats.get("malformed_items", 0) + malformed
    print(f"  posts returned: {len(candidates)} (model skipped {skipped}, malformed {malformed})")
    return candidates


# ============================================================
# LOCAL FILTERING (between Stage 2 and Stage 3)
# ============================================================

def local_clean(candidates: list[Candidate], source_index: SourceIndex,
                seen_keys: set[str], stats: dict) -> list[Candidate]:
    survivors: list[Candidate] = []
    rejects: Counter = Counter()
    kept_stems: list[set[str]] = []
    for cand in candidates:
        reason = None
        if len(cand.text) < MIN_POST_CHARS:
            reason = "too_short"
        elif len(cand.text) > stats["max_chars"]:
            reason = "too_long"
        else:
            reason = find_violation(cand.text) or source_index.copy_reason(cand.text)
        key = dedupe_key(cand.text)
        if reason is None and key in seen_keys:
            reason = "exact_duplicate"
        if reason is None:
            mine = stems(cand.text)
            if any(jaccard(mine, other) >= 0.8 for other in kept_stems):
                reason = "lexical_near_duplicate"
            else:
                kept_stems.append(mine)
        if reason:
            cand.status = f"rejected_local:{reason}"
            rejects[reason] += 1
            continue
        seen_keys.add(key)
        survivors.append(cand)
    stats["local_rejects"] = stats.get("local_rejects", Counter()) + rejects
    return survivors


# ============================================================
# STAGE 3: AI QUALITY SCORING
# ============================================================
# Rubric and threshold are the previous Stage 3's, kept on purpose because it left a healthy
# number of candidates. Concrete changes: (1) batches of SCORE_BATCH_SIZE in parallel instead of
# one call with hundreds (index drift and position bias), (2) the copy-detection duty is removed
# from the prompt since the model was never shown source posts (done locally instead), (3) a
# `formulaic` label and a `topic` label are returned in the same call, and (4) unscored posts are
# retried and then reported instead of vanishing.

SCORING_SYSTEM = """
You are evaluating candidate anonymous university social-app posts.

IMPORTANT:

You are NOT an aggressive rejection filter.

Your job is to SCORE posts.

The goal is to identify:

1. obvious garbage
2. obvious AI-generated generic content
3. unsafe/privacy-violating content
4. believable human posts

A post does NOT need to be interesting.

A post does NOT need to be profound.

A post does NOT need to be perfectly written.

A post does NOT need to generate comments.

A mundane post can be good if it contains a believable human situation.

Posts may be Russian, Kazakh, English or any mix. Casual Kazakh spelling, transliteration and
code-switching are normal and must not be penalized.

Do not punish:

- lowercase writing
- imperfect punctuation
- awkward wording
- slang
- mixed languages
- emotional uncertainty
- short posts
- mundane situations with context
- lack of conclusion

Only score down when there is a real problem.

SCORING:

10 = extremely believable human anonymous post
9 = very believable
8 = strong
7 = good
6 = acceptable and usable
5 = borderline but usable
4 = weak / noticeably generic
3 = clearly poor
2 = very poor
1 = unusable
0 = unsafe or completely unusable

Reject-worthy problems include:

- identifying a private person
- requests for usernames/contact information
- phone numbers
- social handles
- doxxing
- targeted harassment
- threats
- sexual content involving identifiable people
- fake specific university facts (named professors, buildings, rooms, clubs, events, prices, policies, schedules)
- obvious AI poetic writing
- generic motivational content
- generic engagement bait
- completely meaningless filler

IMPORTANT:

Do NOT reject a post merely because it is short.

Do NOT reject a post merely because it is mundane.

Do NOT reject a post merely because it is awkward.

Do NOT reject a post merely because it asks a question.

We are looking for realistic anonymous student behavior,
not polished writing.
"""


def build_scoring_prompt(batch: list[Candidate], references: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(batch))
    reference_block = "\n".join(f"- {r}" for r in references)
    return f"""
REFERENCE: real posts from the platform. They only show what believable looks like (topics and
wording of the candidates will differ). Judge each candidate by how plausibly it could sit among
them. Real posts are addressed to other students, low effort and often blunt. A candidate that
reads like a tidy diary entry, a third-person report, a cozy micro-story, or a note translated
from English is weaker than one that sounds typed in a hurry, however well written it is. Use
the whole scale; most candidates should not get 8 or higher.

{reference_block}

Score every candidate.

Do not rewrite anything.

For every post provide:

- index
- score from 0 to 10
- keep = true or false
- formulaic = true or false
- ai_feel = true or false
- topic (one of: {", ".join(DOMAINS)})
- reason

KEEP RULE:

keep = true when score >= {QUALITY_THRESHOLD}

Exception:

If a post is unsafe, identifying, contains contact information,
or contains fake specific university information,
keep = false regardless of score.

Do NOT be overly strict.

We need a healthy number of approved posts.

If a post is merely "not amazing", keep it.

FORMULAIC LABEL (a label, not a rejection; formulaic posts can still be kept):

formulaic = true when the post follows the stock template
"something happened -> the writer's feelings -> uncertainty -> asking what to do / how to",
or is written in therapy / self-help vocabulary (boundaries, sincerity, unsaid feelings,
"how do I return the small joys"), or reads as a tidy, balanced, grammatical "AI paragraph", a
polished diary entry, or a neat report of a routine event with no point. A short blunt post, a
plain question, or a one-line joke is NOT formulaic.

AI_FEEL LABEL (relative, within this batch): mark ai_feel = true for roughly the quarter of the
candidates that feel MOST machine-written: over-specified, tidy, inventive anecdotes with a neat
ending, product-manager or event-organizer phrasing, translated-sounding wording. This includes a
minor mishap that resolves into a tidy little narrative arc written like a case study (a double
charge that "I wrote to support, waiting to hear back", a last-minute favor that "I agreed to even
though it was unprepared", a mix-up explained step by step to a neat close) - real posts about
this kind of thing are usually blunter and stop before the resolution. Do this even if all of them
are decent; it ranks candidates, it does not reject them.

CANDIDATES:

{numbered}

Return JSON exactly:

{{
    "evaluations": [
        {{
            "index": 1,
            "score": 8,
            "keep": true,
            "formulaic": false,
            "ai_feel": false,
            "topic": "food",
            "reason": "specific human situation"
        }}
    ]
}}
"""


def score_batch(job: tuple) -> dict[int, dict]:
    batch, references = job
    data = call_json(SCORING_SYSTEM, build_scoring_prompt(batch, references), purpose="score", max_tokens=6000)
    evaluations = data.get("evaluations")
    if not isinstance(evaluations, list):
        raise PipelineError("score: response has no 'evaluations' list")
    out: dict[int, dict] = {}
    for item in evaluations:
        if not isinstance(item, dict):
            continue
        index, score = item.get("index"), item.get("score")
        if isinstance(index, int) and 1 <= index <= len(batch) and isinstance(score, (int, float)):
            out[index] = item
    return out


def stage3_score(candidates: list[Candidate], rng: random.Random, stats: dict,
                 exemplars: ExemplarPool) -> list[Candidate]:
    print(f"\nStage 3: AI quality scoring of {len(candidates)} candidates...", flush=True)
    shuffled = candidates[:]
    rng.shuffle(shuffled)  # avoid position bias between types
    unscored = shuffled
    for _ in (1, 2):  # the second pass only re-sends posts the model skipped or mangled
        batches = [unscored[i:i + SCORE_BATCH_SIZE] for i in range(0, len(unscored), SCORE_BATCH_SIZE)]
        jobs = [(batch, rng.sample(exemplars.items, 6)) for batch in batches]
        results = run_parallel("score", score_batch, jobs)
        missing = []
        for batch, evaluations in zip(batches, results):
            for index, cand in enumerate(batch, start=1):
                item = (evaluations or {}).get(index)
                if item is None:
                    missing.append(cand)
                    continue
                cand.score = float(item["score"])
                cand.reason = str(item.get("reason", ""))[:200]
                cand.formulaic = item.get("formulaic") is True
                cand.ai_feel = item.get("ai_feel") is True
                topic = item.get("topic")
                cand.topic = topic if topic in DOMAINS else cand.seed.domain
                keep = item.get("keep") is True and cand.score >= QUALITY_THRESHOLD
                cand.status = "approved" if keep else "rejected_ai"
        unscored = missing
        if not unscored:
            break
    for cand in unscored:
        cand.status = "unscored"
    stats["unscored"] = stats.get("unscored", 0) + len(unscored)
    if unscored:
        print(f"  WARNING: {len(unscored)} posts could not be scored and are excluded.")
    return [c for c in candidates if c.status == "approved"]


# ============================================================
# FEATURES (deterministic structure / language classification)
# ============================================================

PHRASE_PATTERNS = {
    "anyone_else": _rx(r"кто[- ]?(?:нибудь|то)?\s*(?:ещё|еще|тоже)|у кого (?:ещё|еще|тоже)|только у меня|"
                       r"это только я|только я\b|anyone else|is it just me|am i the only|біреу\s+де\b"),
    "why_people": _rx(r"почему (?:люди|все|многие|парни|девушки|девочки|мы)\b|why do (?:people|guys|girls|we|everyone)|"
                      r"why does everyone|неге (?:адамдар|бәрі)"),
    "dont_know_if": _rx(r"не знаю,? (?:если|стоит|правильно|как|что|нужно ли|можно ли|почему)|"
                        r"i don'?t know (?:if|how|what|why|whether)|білмеймін"),
    "what_would_you_do": _rx(r"что бы вы|как бы вы|what would you do|как вы (?:думаете|считаете)|как думаете|"
                             r"а вы как|что думаете|what do you think"),
    "what_to_do": _rx(r"что (?:мне )?делать|как (?:мне )?(?:быть|поступить)|как себя вести|what should i do|"
                      r"what do i do|not sure what to do|не істеу керек|қайттім"),
    "how_to_verb": _rx(r"как (?:мне )?(?:правильно |лучше |спокойно )?(?:спросить|предложить|начать|вернуть|"
                       r"перестать|убедить|справиться|понять|сказать|объяснить|попросить|извиниться|отказать)"),
    "stoit_li": _rx(r"стоит ли|нужно ли"),
}
ADVICE_PATTERNS = {"what_to_do", "how_to_verb"}
ADVICE_EXTRA = _rx(r"посоветуйте|дайте совет|совет\b|any advice|any tips|need advice|кеңес")
# Vocabulary that reads as therapy / self-help / literary rather than typed-in-a-hurry.
TELL_WORDS = _rx(r"недосказан|поверхностн|искренн|навязчив|личные границы|без давления|принадлежн|осознал|"
                 r"воспоминан|как вернуть|странное (?:чувство|ощущение)|хрупк|уязвим|boundaries|vulnerab|"
                 r"authentic|genuine")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
FIRST_PERSON = re.compile(r"\b(?:я|мне|меня|мой|моя|мои|у меня|i|my|me|мен|маған|менің)\b", re.IGNORECASE)


def length_band(length: int) -> int:
    for index, bound in enumerate(LENGTH_BANDS):
        if length <= bound:
            return index
    return len(LENGTH_BANDS)


def extract_features(cand: Candidate) -> dict:
    text = cand.text
    ws = words(text)
    hits = {name for name, pattern in PHRASE_PATTERNS.items() if pattern.search(text)}
    is_question = "?" in text
    advice = bool(hits & ADVICE_PATTERNS) or bool(ADVICE_EXTRA.search(text)) or \
        ("dont_know_if" in hits and is_question)
    sentences = [s for s in re.split(r"[.!?…]+\s*", text) if s.strip()]
    if text.endswith("?"):
        ending = "q"
    elif text.endswith(("...", "…")):
        ending = "ellipsis"
    elif EMOJI_RE.search(text[-2:]):
        ending = "emoji"
    elif text[-1:].isalnum() or text[-1:] in ")":
        ending = "bare"
    else:
        ending = "stop"
    tells = len(TELL_WORDS.findall(text)) + (1 if len(text) < 150 and text.count(",") >= 3 else 0)
    return {
        "length": len(text),
        "lenband": length_band(len(text)),
        "is_question": is_question,
        "advice": advice,
        "patterns": hits,
        "opener1": (ws[0][:4] if ws else "?"),
        "opener2": " ".join(w[:4] for w in ws[:2]),
        "ending": ending,
        "casing": "lower" if text[:1].islower() else "upper",
        "shape": f"s{min(len(sentences), 3)}_{'fp' if FIRST_PERSON.search(text) else 'np'}",
        "tells": tells,
        "emoji": bool(EMOJI_RE.search(text)),
    }


def prepare_candidates(cands: list[Candidate]) -> str:
    """Detect language, extract features, embed. Returns the similarity backend in use."""
    for cand in cands:
        cand.language = detect_language(cand.text)
        cand.features = extract_features(cand)
    vectors = embed_texts([c.text for c in cands])
    if vectors is None:
        return "lexical"
    for cand, vector in zip(cands, vectors):
        cand.embedding = vector
    return "embedding"


def similarity(a: Candidate, b: Candidate, backend: str) -> float:
    if backend == "embedding" and a.embedding and b.embedding:
        return dot(a.embedding, b.embedding)
    return jaccard(stems(a.text), stems(b.text))


# ============================================================
# STAGE 4: DIVERSITY-AWARE SELECTION
# ============================================================
# Objective over a set S of FINAL_COUNT posts:
#   + W_QUALITY * quality                       (Stage 3 score, mildly weighted)
#   + sum_attr w * sum_values sqrt(count)       (coverage with diminishing returns: a new value
#                                                 pays more than a repeat, but nothing is a quota)
#   - W_SIM * pairwise meaning overlap          (soft above `soft`, forbidden above `hard`)
#   - excess advice-asking / questions / repeated stock phrases / AI-tell vocabulary
# Built by greedy add, then improved by single-swap local search. Hard caps only block the
# extremes; if they leave fewer than FINAL_COUNT posts the caller runs recovery or errors out.

@dataclass
class SelectionContext:
    cands: list[Candidate]
    backend: str
    soft: float
    hard: float
    sim: list[list[float]]
    length_prior: list[float]  # per length band, 0..1, from the real corpus
    targets: dict[str, dict]   # {"casing": {"lower": target_count, ...}, "emoji": {...}}

    def attr(self, cand: Candidate, name: str):
        f = cand.features
        return {"type": cand.post_type, "topic": cand.topic, "lang": cand.language,
                "lenband": f["lenband"], "opener": f["opener1"], "ending": f["ending"],
                "shape": f["shape"]}[name]


def build_context(cands: list[Candidate], backend: str, length_prior: list[float],
                  targets: dict[str, dict]) -> SelectionContext:
    n = len(cands)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim[i][j] = sim[j][i] = similarity(cands[i], cands[j], backend)
    thresholds = SIM_THRESHOLDS[backend]
    return SelectionContext(cands, backend, thresholds["soft"], thresholds["hard"], sim, length_prior, targets)


def habit_targets(final_count: int, habit_rates: dict[str, float]) -> dict[str, dict]:
    """Soft target COUNTS (real corpus rate x final size) for binary typing habits.

    Used with target_bonus(), a concave function of how many are already chosen: below target it
    keeps paying (declining) rewards, at target it is flat, past target it costs. That pulls the
    final proportion toward the real one without ever hard-capping it.
    """
    lower = round(final_count * habit_rates["lowercase"])
    emoji = round(final_count * habit_rates["emoji"])
    return {
        "casing": {"lower": lower, "upper": final_count - lower},
        "emoji": {True: emoji, False: final_count - emoji},
    }


def target_bonus(count: int, target: int) -> float:
    """Concave reward: full marginal value while under target, zero at target, negative past it."""
    if count < target:
        return 1.0 - count / max(target, 1) * 0.4  # 1.0 -> 0.6 as it approaches target
    return -0.3 * (count - target)


def corpus_length_prior(corpus_posts: list[dict]) -> list[float]:
    """Relative frequency of each length band among real posts, scaled so the commonest is 1.

    Used as a gentle tie-breaker so the final set is not systematically longer than what people
    really post. It is not a quota: the coverage rewards still spread the set across bands."""
    counts = Counter(length_band(len(normalize(p.get("text", "")))) for p in corpus_posts)
    top = max(counts.values()) if counts else 1
    return [max(0.25, counts.get(i, 0) / top) for i in range(len(LENGTH_BANDS) + 1)]


def question_soft_limit(profile: dict) -> int:
    share = profile.get("questions", {}).get("question_percentage", 42) / 100
    return round(FINAL_COUNT * (share + 0.05))


def objective(ctx: SelectionContext, chosen: list[int], question_limit: int) -> float:
    if not chosen:
        return 0.0
    cands = [ctx.cands[i] for i in chosen]
    total = 0.0
    for cand in cands:
        quality = min(1.0, max(0.0, (cand.score - 4) / 6))
        total += W_QUALITY * quality
        total += W_LEN_PRIOR * ctx.length_prior[cand.features["lenband"]]
        total -= W_TELL * min(cand.features["tells"], 3)
        total -= W_FORMULAIC if cand.formulaic else 0.0
        total -= W_AI_FEEL if cand.ai_feel else 0.0
    for name, weight in COVERAGE_WEIGHTS.items():
        counts = Counter(ctx.attr(c, name) for c in cands)
        total += weight * sum(math.sqrt(v) for v in counts.values())
    for name, weight in W_TARGET.items():
        counts = Counter((c.features["casing"] if name == "casing" else c.features["emoji"]) for c in cands)
        for value, target in ctx.targets[name].items():
            total += weight * target_bonus(counts.get(value, 0), target)
    span = max(ctx.hard - ctx.soft, 1e-6)
    for a in range(len(chosen)):
        for b in range(a + 1, len(chosen)):
            s = ctx.sim[chosen[a]][chosen[b]]
            if s > ctx.soft:
                total -= W_SIM * (s - ctx.soft) / span
    total -= W_QUESTION_EXCESS * max(0, sum(c.features["is_question"] for c in cands) - question_limit)
    total -= W_ADVICE_EXCESS * max(0, sum(c.features["advice"] for c in cands) - 2)
    pattern_counts = Counter(p for c in cands for p in c.features["patterns"])
    total -= W_PATTERN_REPEAT * sum(max(0, v - 1) for v in pattern_counts.values())
    return total


def blocked_reason(ctx: SelectionContext, chosen: list[int], new: int) -> Optional[str]:
    """Why `new` cannot join `chosen` at all (hard constraints only)."""
    cand = ctx.cands[new]
    members = [ctx.cands[i] for i in chosen]
    for i in chosen:
        if ctx.sim[i][new] >= ctx.hard:
            return "same_meaning"
    if any(m.seed.id == cand.seed.id for m in members):
        return "same_seed"
    if any(dedupe_key(m.seed.theme) == dedupe_key(cand.seed.theme) for m in members):
        return "same_theme"
    f = cand.features
    if sum(m.topic == cand.topic for m in members) >= HARD_CAPS["topic"]:
        return "topic_cap"
    if sum(m.post_type == cand.post_type for m in members) >= HARD_CAPS["type"]:
        return "type_cap"
    if f["advice"] and sum(m.features["advice"] for m in members) >= HARD_CAPS["advice"]:
        return "advice_cap"
    if sum(m.features["opener1"] == f["opener1"] for m in members) >= HARD_CAPS["opener1"]:
        return "opener_cap"
    if f["opener2"].count(" ") and sum(m.features["opener2"] == f["opener2"] for m in members) >= HARD_CAPS["opener2"]:
        return "opener_cap"
    return None


def select_final(cands: list[Candidate], backend: str, question_limit: int,
                 length_prior: list[float], targets: dict[str, dict],
                 target: int = FINAL_COUNT) -> tuple[list[Candidate], Counter]:
    """Returns the chosen posts (possibly fewer than `target`) and why others were blocked."""
    ctx = build_context(cands, backend, length_prior, targets)
    chosen: list[int] = []
    blocked: Counter = Counter()

    while len(chosen) < target:
        base = objective(ctx, chosen, question_limit)
        best, best_gain = None, -math.inf
        stall = Counter()
        for i in range(len(cands)):
            if i in chosen:
                continue
            reason = blocked_reason(ctx, chosen, i)
            if reason:
                stall[reason] += 1
                continue
            gain = objective(ctx, chosen + [i], question_limit) - base
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            blocked = stall
            break
        chosen.append(best)

    # Local search: for each position, replace the post with the best alternative if that raises
    # the objective. Repeats until a full pass changes nothing.
    if len(chosen) == target:
        for _ in range(5):
            changed = False
            for position in range(len(chosen)):
                rest = chosen[:position] + chosen[position + 1:]
                best_i, best_value = chosen[position], objective(ctx, chosen, question_limit)
                for i in range(len(cands)):
                    if i in chosen or blocked_reason(ctx, rest, i):
                        continue
                    value = objective(ctx, rest + [i], question_limit)
                    if value > best_value + 1e-9:
                        best_i, best_value = i, value
                if best_i != chosen[position]:
                    chosen[position] = best_i
                    changed = True
            if not changed:
                break

    return [cands[i] for i in chosen], blocked


# ============================================================
# FINAL VALIDATION
# ============================================================

WARN_PHRASES = {
    "что делать": _rx(r"что (?:мне )?делать"),
    "кто-нибудь": _rx(r"кто[- ]?нибудь"),
    "почему люди": _rx(r"почему (?:люди|все)"),
    "не знаю": _rx(r"\bне знаю\b"),
    "как быть": _rx(r"как (?:мне )?быть"),
    "has anyone / anyone else": _rx(r"has anyone|anyone else"),
    "вдруг": _rx(r"\bвдруг\b"),
}


def validate_final(final: list[Candidate], backend: str, profile: dict,
                   max_chars_soft: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    texts = [c.text for c in final]

    if len(final) != FINAL_COUNT:
        errors.append(f"expected {FINAL_COUNT} posts, have {len(final)}")
    for cand in final:
        violation = find_violation(cand.text)
        if violation:
            errors.append(f"post {cand.id} fails safety check '{violation}': {cand.text}")
    if len({dedupe_key(t) for t in texts}) != len(texts):
        errors.append("exact duplicate posts in final set")
    hard = SIM_THRESHOLDS[backend]["hard"]
    for i in range(len(final)):
        for j in range(i + 1, len(final)):
            s = similarity(final[i], final[j], backend)
            if s >= hard:
                errors.append(f"near-duplicate pair (sim {s:.2f}): {final[i].text!r} / {final[j].text!r}")
            elif s >= (SIM_THRESHOLDS[backend]["soft"] + hard) / 2:
                warnings.append(f"similar pair (sim {s:.2f}): {final[i].text!r} / {final[j].text!r}")
            if jaccard(stems(texts[i]), stems(texts[j])) >= 0.5:
                warnings.append(f"high lexical overlap: {texts[i]!r} / {texts[j]!r}")

    n = len(final)
    questions = sum(c.features["is_question"] for c in final)
    if questions > question_soft_limit(profile):
        warnings.append(f"{questions}/{n} posts are questions")
    advice = sum(c.features["advice"] for c in final)
    if advice > 2:
        warnings.append(f"{advice}/{n} posts ask for advice ('what should I do' style)")

    for label, counter in (("opening word", Counter(c.features["opener1"] for c in final)),
                           ("opening two words", Counter(c.features["opener2"] for c in final if " " in c.features["opener2"]))):
        for value, count in counter.items():
            if count >= 3:
                warnings.append(f"{count}/{n} posts share the {label} '{value}'")
    for label, counter in (("topic", Counter(c.topic for c in final)),
                           ("post type", Counter(c.post_type for c in final)),
                           ("language", Counter(c.language for c in final)),
                           ("shape", Counter(c.features["shape"] for c in final))):
        value, count = counter.most_common(1)[0]
        limit = {"language": 14, "shape": 9}.get(label, 6)
        if count >= limit:
            warnings.append(f"{count}/{n} posts share the same {label}: {value}")

    lengths = [len(t) for t in texts]
    bands = Counter(c.features["lenband"] for c in final)
    if max(bands.values()) >= 0.6 * n:
        warnings.append(f"{max(bands.values())}/{n} posts fall in one length band")
    if len(lengths) > 1 and statistics.pstdev([math.log(x) for x in lengths]) < 0.45:
        warnings.append("length distribution is suspiciously narrow")
    if max(lengths) > max_chars_soft:
        warnings.append(f"longest post is {max(lengths)} chars (soft limit {max_chars_soft})")

    for phrase, pattern in WARN_PHRASES.items():
        count = sum(bool(pattern.search(t)) for t in texts)
        if count >= 3:
            warnings.append(f"phrase '{phrase}' appears in {count}/{n} posts")
    term_counts = Counter(s for t in texts for s in stems(t) if len(s) >= 5)
    for term, count in term_counts.most_common(3):
        if count >= 4:
            warnings.append(f"word stem '{term}' appears in {count}/{n} posts")

    lower_share = sum(c.features["casing"] == "lower" for c in final) / n
    profile_lower = profile.get("punctuation", {}).get("percentages", {}).get("lowercase_start", 22) / 100
    if lower_share > profile_lower + 0.25:
        warnings.append(f"{lower_share:.0%} of posts start lowercase (reference corpus: {profile_lower:.0%})")
    emoji_share = sum(c.features["emoji"] for c in final) / n
    profile_emoji = profile.get("emoji", {}).get("percentage_with_emoji", 15) / 100
    if emoji_share > profile_emoji + 0.25:
        warnings.append(f"{emoji_share:.0%} of posts contain emoji (reference corpus: {profile_emoji:.0%})")
    return errors, warnings


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_counter(title: str, counter: Counter, total: Optional[int] = None) -> None:
    print(f"  {title}:")
    for key, count in counter.most_common():
        print(f"    {key}: {count}" + (f" ({count / total:.0%})" if total else ""))


def print_score_distribution(cands: list[Candidate]) -> None:
    scored = Counter(int(c.score) for c in cands if c.score is not None)
    print("  score distribution: " + ", ".join(f"{s}:{scored[s]}" for s in sorted(scored, reverse=True)))


def print_final_report(final: list[Candidate], warnings: list[str]) -> None:
    lengths = [len(c.text) for c in final]
    n = len(final)
    print("\nFinal 20:\n")
    for index, cand in enumerate(final, start=1):
        print(f"{index:>2}. {cand.text}")
        print(f"      [{cand.post_type} | {cand.topic} | {cand.language} | {len(cand.text)} chars | score {cand.score:.0f}"
              f"{' | formulaic' if cand.formulaic else ''}{' | ai_feel' if cand.ai_feel else ''}]")
    print("\nFinal statistics:")
    labels = ["<=30", "31-60", "61-100", "101-150", "151-220", "221-330", ">330"]
    bands = Counter(c.features["lenband"] for c in final)
    print("  length distribution: " + ", ".join(f"{labels[i]}:{bands.get(i, 0)}" for i in range(len(labels))))
    print(f"  average / median length: {sum(lengths) / n:.1f} / {statistics.median(lengths):.0f}")
    print(f"  shortest: {min(lengths)}   longest: {max(lengths)}")
    print_counter("post types", Counter(c.post_type for c in final))
    print_counter("topics", Counter(c.topic for c in final))
    print_counter("languages (detected)", Counter(c.language for c in final))
    print(f"  questions: {sum(c.features['is_question'] for c in final)}/{n}   "
          f"advice-seeking: {sum(c.features['advice'] for c in final)}/{n}   "
          f"formulaic: {sum(c.formulaic for c in final)}/{n}   ai_feel: {sum(c.ai_feel for c in final)}/{n}")
    kz_count = sum(c.language in ("kz", "ru_kz") for c in final)
    if kz_count:
        print(f"\nNOTE: {kz_count}/{n} posts contain Kazakh. Kazakh output has not been reviewed by "
              "a native speaker in this pipeline - have someone check it before publishing.")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("\nNo diversity warnings.")


def write_debug_file(records: list[Candidate], stats: dict) -> None:
    payload = {
        "stats": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()},
        "candidates": [
            {"id": c.id, "round": c.round, "status": c.status, "type": c.post_type, "score": c.score,
             "formulaic": c.formulaic, "ai_feel": c.ai_feel, "topic": c.topic or c.seed.domain, "seed_theme": c.seed.theme,
             "seed_kind": c.seed.kind, "seed_subject": c.seed.subject, "assigned_language": c.assigned_language,
             "detected_language": c.language, "text": c.text}
            for c in records
        ],
    }
    with open(DEBUG_FILE, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=1)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    rng = random.Random(int(os.getenv("UNITEE_SEED", "0")) or None)

    print("Loading configuration files...")
    profile = load_json(STYLE_FILE)
    rules = load_json(RULES_FILE)
    strategy = load_json(STRATEGY_FILE)
    hard_limits = build_hard_limits(rules)
    corpus_posts, source_index = load_corpus()

    flagged = load_json(BASE_DIR / "flagged_telegram_posts.json", required=False) or {}
    flagged_texts = {normalize(p.get("text", "")) for p in flagged.get("posts", []) if isinstance(p, dict)}
    exemplars = ExemplarPool(corpus_posts, flagged_texts)
    if len(exemplars.items) < 20:
        raise PipelineError(f"Only {len(exemplars.items)} safe exemplar posts available; need at least 20.")
    language_weights = derive_language_weights(corpus_posts)
    length_prior = corpus_length_prior(corpus_posts)
    habit_rates = {  # per-post typing habits, at the rates the style profile measured
        "lowercase": profile.get("punctuation", {}).get("percentages", {}).get("lowercase_start", 22) / 100,
        "emoji": profile.get("emoji", {}).get("percentage_with_emoji", 15) / 100,
    }

    soft_max = int(strategy.get("length", {}).get("long_post_max_characters", 500))
    stats: dict = {"max_chars": int(soft_max * 1.1)}
    print(f"  corpus: {len(corpus_posts)} posts, {len(exemplars.items)} usable as style exemplars")
    print("  language mix for generation: " + ", ".join(f"{k} {v:.0%}" for k, v in language_weights.items()))

    all_records: list[Candidate] = []
    approved: list[Candidate] = []
    seen_keys: set[str] = set()
    used_themes: list[str] = []
    next_seed_id = next_cand_id = 0
    backend = "embedding"
    final: list[Candidate] = []
    blocked: Counter = Counter()

    try:
        for round_number in range(1, MAX_ROUNDS + 1):
            scale = 1.0 if round_number == 1 else 0.7
            if round_number > 1:
                print(f"\n=== RECOVERY ROUND {round_number}: only {len(final)} selectable posts after round "
                      f"{round_number - 1} (blocked by: {dict(blocked)}). Generating fresh seeds and candidates. ===")

            # ---- Stage 1
            raw_seeds = stage1_seeds(rng, hard_limits, start_id=next_seed_id,
                                     avoid_themes=used_themes, stats=stats)
            next_seed_id += len(raw_seeds) + 1
            seeds, backend = dedupe_seeds(raw_seeds, stats)
            used_themes += [s.theme for s in seeds]
            print(f"  unique seeds after semantic de-duplication ({backend}): {len(seeds)}")
            print_counter("seeds per kind", Counter(s.kind for s in seeds))
            print_counter("seeds per domain", Counter(s.domain for s in seeds))
            if len(seeds) < FINAL_COUNT * 4:
                raise PipelineError(f"Only {len(seeds)} unique scenarios generated; need at least {FINAL_COUNT * 4}.")

            # ---- Stage 2
            batches = plan_batches(seeds, rng, language_weights, scale, stats, habit_rates)
            candidates = stage2_generate(batches, exemplars, rng, hard_limits,
                                         round_number, next_cand_id, stats)
            next_cand_id += len(candidates) + 1
            all_records += candidates
            print_counter("candidates by post type", Counter(c.post_type for c in candidates))
            survivors = local_clean(candidates, source_index, seen_keys, stats)
            print(f"  candidates after local filtering: {len(survivors)}/{len(candidates)}")
            print(f"  removed locally: {dict(stats['local_rejects'])}")
            if len(survivors) < FINAL_COUNT * 2:
                raise PipelineError(f"Only {len(survivors)} candidates survived local filtering; "
                                    f"need at least {FINAL_COUNT * 2}.")

            # ---- Stage 3
            newly_approved = stage3_score(survivors, rng, stats, exemplars)
            print(f"  approved: {len(newly_approved)}   rejected: "
                  f"{sum(c.status == 'rejected_ai' for c in survivors)}   unscored: {stats.get('unscored', 0)}")
            print_score_distribution(survivors)
            approved += newly_approved

            # ---- Stage 4
            print(f"\nStage 4: diversity-aware selection from {len(approved)} approved posts...", flush=True)
            backend = prepare_candidates(approved)
            final, blocked = select_final(approved, backend, question_soft_limit(profile),
                                          length_prior, habit_targets(FINAL_COUNT, habit_rates))
            print(f"  selected {len(final)}/{FINAL_COUNT} (similarity backend: {backend})")
            if len(final) == FINAL_COUNT:
                break

        if len(final) < FINAL_COUNT:
            raise PipelineError(
                f"only {len(final)} sufficiently diverse approved posts available "
                f"(approved pool {len(approved)}, candidates blocked by: {dict(blocked)})")

        errors, warnings = validate_final(final, backend, profile, soft_max)
        for cand in final:
            cand.status = "selected"
        if errors:
            raise PipelineError("final validation failed:\n  " + "\n  ".join(errors))

        texts = [c.text for c in final]
        rng.shuffle(texts)  # output order should not reveal post type
        tmp_path = OUTPUT_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(texts, file, ensure_ascii=False, indent=2)
        tmp_path.replace(OUTPUT_FILE)

        print_final_report(final, warnings)
        print(f"\nSaved to: {OUTPUT_FILE}")
        return 0
    finally:
        if all_records:
            write_debug_file(all_records, stats)
            print(f"Debug info: {DEBUG_FILE.name}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
