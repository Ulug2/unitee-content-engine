import json
import re
from collections import Counter

INPUT_FILE = "raw_telegram_posts.json"

SAFE_FILE = "safe_style_posts.json"
REVIEW_FILE = "review_posts.json"
EXCLUDE_FILE = "excluded_posts.json"


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_text(text):
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_contact_information(text):
    patterns = [
        # URLs
        r"https?://\S+",
        r"www\.\S+",

        # Telegram / Instagram / social usernames
        r"\bt\.me/\S+",
        r"@[A-Za-z0-9_]{4,}",

        # Email
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",

        # Phone numbers
        r"\+?\d[\d\s().-]{7,}\d",
    ]

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


# ============================================================
# PERSONAL IDENTIFICATION
# ============================================================

def has_identification_request(text):
    lowered = text.lower()

    patterns = [
        # Social media / contact requests
        r"\b(инст|инста|инстаграм|тг|телеграм|юз|username)\b",
        r"\b(дай|дайте|можно|скинь|скиньте)\b.*\b(инст|тг|юз)\b",

        # Asking for information about a person
        r"\bкто знает\b",
        r"\bкто-нибудь знает\b",
        r"\bможно инфу\b",
        r"\bдайте инфу\b",
        r"\bинфу про\b",

        # Relationship status
        r"\bесть (ли )?(у него|у неё|у нее) девушка\b",
        r"\bесть (ли )?(у него|у неё|у нее) парень\b",
        r"\bсвободен\b",
        r"\bсвободна\b",
        r"\bsingle\b",

        # Describing a specific person
        r"\bпарень в\b",
        r"\bдевушка в\b",
        r"\bдевочка в\b",
        r"\bмальчик в\b",
        r"\bкьюти в\b",

        # Common identification details
        r"\b\d+\s*(курс|курса)\b.*\b(механик|эконом|экон|юрфак|math|physics|физик|engineering|engineering)\b",
    ]

    return any(re.search(pattern, lowered) for pattern in patterns)


def appears_to_target_a_person(text):
    lowered = text.lower()

    person_words = [
        "парень",
        "девушка",
        "девочка",
        "мальчик",
        "краш",
        "кьюти",
        "аниме тян",
    ]

    targeting_words = [
        "кто",
        "кто-нибудь",
        "инфу",
        "юз",
        "инст",
        "тг",
        "свободен",
        "свободна",
        "есть девушка",
        "есть парень",
    ]

    has_person_word = any(word in lowered for word in person_words)
    has_targeting_word = any(word in lowered for word in targeting_words)

    return has_person_word and has_targeting_word


# ============================================================
# SENSITIVE CONTENT
# ============================================================

def has_sensitive_content(text):
    lowered = text.lower()

    keywords = [
        # Self-harm / suicide
        "суицид",
        "самоуб",
        "убить себя",
        "умереть",
        "хочу умереть",
        "suicide",
        "kill myself",

        # Sexual / explicit
        "nudes",
        "голый",
        "голая",
        "изнасил",
        "насил",
        "rape",

        # Sexual solicitation
        "секс",
        "sex",
    ]

    return any(keyword in lowered for keyword in keywords)


# ============================================================
# HARASSMENT / TARGETED ABUSE
# ============================================================

def has_targeted_harassment(text):
    lowered = text.lower()

    harassment_patterns = [
        r"\bидиот\b",
        r"\bдебил\b",
        r"\bтупой\b",
        r"\bтупая\b",
        r"\bмразь\b",
        r"\bурод\b",
        r"\bшлюх",
        r"\bпроститут",
    ]

    return any(re.search(pattern, lowered) for pattern in harassment_patterns)


# ============================================================
# ADVERTISEMENT / EXTERNAL PROMOTION
# ============================================================

def looks_like_advertisement(text):
    lowered = text.lower()

    ad_patterns = [
        "ваканс",
        "ищем сотрудников",
        "ищем преподавател",
        "на работу",
        "для подробной информации",
        "пишите в whatsapp",
        "подписывайтесь",
        "скидка",
        "продам",
        "куплю",
        "заказ",
    ]

    return any(pattern in lowered for pattern in ad_patterns)


# ============================================================
# LOW QUALITY
# ============================================================

def is_too_short(text):
    # Character threshold
    if len(text.strip()) < 15:
        return True

    # Word threshold
    words = text.split()

    if len(words) < 3:
        return True

    return False


def looks_like_noise(text):
    # Extremely repetitive characters
    if re.search(r"(.)\1{7,}", text):
        return True

    # Mostly punctuation / symbols
    alphanumeric = sum(char.isalnum() for char in text)

    if len(text) > 0 and alphanumeric / len(text) < 0.35:
        return True

    return False


# ============================================================
# STYLE VALUE
# ============================================================

def calculate_style_value(text):
    """
    Estimate whether a post contains useful information about
    student communication style.
    """

    score = 0

    words = text.split()

    # Natural conversational length
    if 15 <= len(text) <= 500:
        score += 2

    # Question
    if "?" in text:
        score += 2

    # Emoji
    if re.search(r"[\U0001F300-\U0001FAFF]", text):
        score += 1

    # Mixed language
    has_cyrillic = bool(re.search(r"[А-Яа-яӘәІіҢңҒғҚқӨөҰұҮүҺһ]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))

    if has_cyrillic and has_latin:
        score += 2

    # Casual punctuation / formatting
    if "..." in text or "😭" in text or "😂" in text or "🤣" in text:
        score += 1

    # Conversational first/second person
    conversational_words = [
        "я",
        "мне",
        "мен",
        "маған",
        "сен",
        "ты",
        "вы",
        "мы",
        "біз",
    ]

    lowered_words = set(word.lower().strip(".,!?") for word in words)

    if lowered_words.intersection(conversational_words):
        score += 1

    return score


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_post(text):
    reasons = []

    if is_too_short(text):
        reasons.append("too_short")

    if looks_like_noise(text):
        reasons.append("noise")

    if has_contact_information(text):
        reasons.append("contact_information")

    if has_identification_request(text):
        reasons.append("personal_identification_request")

    if appears_to_target_a_person(text):
        reasons.append("targeted_person_reference")

    if has_sensitive_content(text):
        reasons.append("sensitive_content")

    if has_targeted_harassment(text):
        reasons.append("targeted_harassment")

    if looks_like_advertisement(text):
        reasons.append("advertisement")

    # Hard exclusions
    hard_exclusion_reasons = {
        "contact_information",
        "personal_identification_request",
        "targeted_person_reference",
        "sensitive_content",
        "targeted_harassment",
        "advertisement",
        "noise",
    }

    if any(reason in hard_exclusion_reasons for reason in reasons):
        return "EXCLUDE", reasons

    # Short posts aren't necessarily dangerous,
    # but they're usually not useful enough for style extraction.
    if "too_short" in reasons:
        return "EXCLUDE", reasons

    # Style score
    style_score = calculate_style_value(text)

    if style_score >= 2:
        return "SAFE_STYLE_REFERENCE", reasons

    return "REVIEW", reasons


# ============================================================
# MAIN
# ============================================================

def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    safe_posts = []
    review_posts = []
    excluded_posts = []

    classification_counts = Counter()
    exclusion_reasons = Counter()

    for post in data["posts"]:

        original_text = post.get("text", "")
        text = normalize_text(original_text)

        classification, reasons = classify_post(text)

        classification_counts[classification] += 1

        for reason in reasons:
            exclusion_reasons[reason] += 1

        result = {
            "source": post.get("source"),
            "text": text,
            "classification": classification,
            "flags": reasons,
            "style_score": calculate_style_value(text),
        }

        if classification == "SAFE_STYLE_REFERENCE":
            safe_posts.append(result)

        elif classification == "REVIEW":
            review_posts.append(result)

        else:
            excluded_posts.append(result)

    # Save safe dataset
    with open(SAFE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "total_posts": len(safe_posts),
                "posts": safe_posts,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Save review dataset
    with open(REVIEW_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "total_posts": len(review_posts),
                "posts": review_posts,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Save excluded dataset
    with open(EXCLUDE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "total_posts": len(excluded_posts),
                "posts": excluded_posts,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # REPORT
    # ========================================================

    print("=" * 60)
    print("UNITEe TELEGRAM DATASET V2")
    print("=" * 60)

    print(f"\nOriginal posts: {len(data['posts'])}")

    print("\nClassification:")
    for classification, count in classification_counts.most_common():
        print(f"  {classification}: {count}")

    print("\nFlag/reason counts:")
    for reason, count in exclusion_reasons.most_common():
        print(f"  {reason}: {count}")

    print("\nFiles created:")
    print(f"  {SAFE_FILE}")
    print(f"  {REVIEW_FILE}")
    print(f"  {EXCLUDE_FILE}")

    print("\nDone!")


if __name__ == "__main__":
    main()
