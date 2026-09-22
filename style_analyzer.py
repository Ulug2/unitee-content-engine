import json
import re
from collections import Counter
from statistics import mean, median

INPUT_FILE = "safe_style_posts.json"
OUTPUT_FILE = "unitee_style_profile.json"


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):
    has_cyrillic = bool(
        re.search(r"[А-Яа-яӘәІіҢңҒғҚқӨөҰұҮүҺһ]", text)
    )

    has_latin = bool(re.search(r"[A-Za-z]", text))

    if has_cyrillic and has_latin:
        return "mixed"

    if has_cyrillic:
        return "cyrillic"

    if has_latin:
        return "latin"

    return "other"


# ============================================================
# POST FORMAT
# ============================================================

def detect_format(text):
    stripped = text.strip()
    words = stripped.split()
    word_count = len(words)

    if "?" in stripped:
        if word_count <= 20:
            return "short_question"
        return "question"

    if word_count <= 8:
        return "short_statement"

    if word_count >= 80:
        return "long_story"

    return "general_statement"


# ============================================================
# EMOJIS
# ============================================================

def extract_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "]+"
    )

    return emoji_pattern.findall(text)


# ============================================================
# PUNCTUATION
# ============================================================

def punctuation_features(text):
    return {
        "question_mark": "?" in text,
        "exclamation_mark": "!" in text,
        "ellipsis": "..." in text,
        "multiple_exclamation": "!!" in text,
        "multiple_question": "??" in text,
        "lowercase_start": bool(
            text and text[0].isalpha() and text[0].islower()
        ),
    }


# ============================================================
# CASUAL LANGUAGE INDICATORS
# ============================================================

CASUAL_WORDS = [
    "пж",
    "пжж",
    "пжпж",
    "ко",
    "го",
    "краш",
    "лол",
    "хз",
    "имхо",
    "вайб",
    "вайба",
    "чел",
    "челик",
    "ребят",
    "достар",
    "бля",
    "блять",
    "ебать",
    "wtf",
    "wth",
    "lol",
    "pls",
    "bro",
    "lowkey",
    "literally",
]


def count_casual_indicators(text):
    lowered = text.lower()
    found = []

    for word in CASUAL_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", lowered):
            found.append(word)

    return found


# ============================================================
# LANGUAGE MIXING
# ============================================================

def detect_mixing(text):
    lowered = text.lower()

    has_russian_cyrillic = bool(
        re.search(r"[а-яё]", lowered)
    )

    kazakh_chars = bool(
        re.search(r"[әіңғқөұүһі]", lowered)
    )

    has_english = bool(
        re.search(r"\b(the|is|are|you|what|why|pls|please|girl|boy|single|cute|literally|lowkey)\b", lowered)
    )

    return {
        "russian_indicators": has_russian_cyrillic,
        "kazakh_indicators": kazakh_chars,
        "english_indicators": has_english,
    }


# ============================================================
# QUESTION ANALYSIS
# ============================================================

def question_type(text):
    lowered = text.lower()

    if "?" not in text:
        return "not_question"

    question_starters = {
        "кто": "who",
        "что": "what",
        "где": "where",
        "когда": "when",
        "почему": "why",
        "как": "how",
        "можно": "request",
        "есть": "yes_no",
        "does": "yes_no",
        "is": "yes_no",
        "are": "yes_no",
        "what": "what",
        "why": "why",
        "how": "how",
        "where": "where",
        "калай": "how",
        "қалай": "how",
        "кім": "who",
        "не": "what",
    }

    first_word = re.sub(
        r"[^a-zа-яёәіңғқөұүһі]",
        "",
        lowered.split()[0]
    ) if lowered.split() else ""

    return question_starters.get(first_word, "other_question")


# ============================================================
# PERSONAL LANGUAGE
# ============================================================

def detect_personal_language(text):
    lowered = text.lower()

    first_person_patterns = [
        r"\bя\b",
        r"\bмне\b",
        r"\bмой\b",
        r"\bмоя\b",
        r"\bмы\b",
        r"\bмен\b",
        r"\bмаған\b",
        r"\bменің\b",
        r"\bi\b",
        r"\bmy\b",
        r"\bme\b",
    ]

    second_person_patterns = [
        r"\bты\b",
        r"\bтебе\b",
        r"\bтвой\b",
        r"\bвы\b",
        r"\bвам\b",
        r"\bsen\b",
        r"\bсіз\b",
        r"\byou\b",
        r"\byour\b",
    ]

    first_person = any(
        re.search(pattern, lowered)
        for pattern in first_person_patterns
    )

    second_person = any(
        re.search(pattern, lowered)
        for pattern in second_person_patterns
    )

    return {
        "first_person": first_person,
        "second_person": second_person,
    }


# ============================================================
# TOPIC DETECTION
# ============================================================

TOPIC_KEYWORDS = {

    "campus": [
        "кампус",
        "campus",
        "универ",
        "университет",
        "уни",
        "универс",
        "ну",
        "sdu",
    ],

    "classes": [
        "пара",
        "пары",
        "лекция",
        "лекции",
        "семинар",
        "класс",
        "class",
        "lecture",
        "exam",
        "экзамен",
        "экзамены",
        "сессия",
        "домашка",
        "дз",
    ],

    "professors": [
        "препод",
        "преподователь",
        "профессор",
        "учитель",
        "teacher",
        "professor",
    ],

    "studying": [
        "учеба",
        "учиться",
        "учусь",
        "учебу",
        "study",
        "studying",
        "учёба",
    ],

    "clubs": [
        "клуб",
        "клубы",
        "club",
        "организация",
        "organization",
    ],

    "food": [
        "еда",
        "есть",
        "поесть",
        "столов",
        "кафе",
        "ресторан",
        "кофе",
        "food",
        "dining",
    ],

    "relationships": [
        "девушка",
        "парень",
        "отношения",
        "любовь",
        "краш",
        "dating",
        "relationship",
        "crush",
    ],

    "friendship": [
        "друг",
        "друзья",
        "дружба",
        "подруга",
        "friend",
        "friends",
    ],

    "events": [
        "ивент",
        "ивенты",
        "мероприятие",
        "event",
        "концерт",
        "вечеринка",
        "party",
    ],

    "housing": [
        "общага",
        "общежитие",
        "дорм",
        "roommate",
        "комната",
        "жилье",
        "жильё",
        "dorm",
    ],

    "career": [
        "работа",
        "работать",
        "стажировка",
        "интерн",
        "карьера",
        "работу",
        "job",
        "internship",
        "career",
    ],

    "social_life": [
        "гулять",
        "гуляли",
        "встреча",
        "встретиться",
        "выходные",
        "weekend",
        "тусовка",
        "тусить",
    ],
}


def detect_topics(text):
    lowered = text.lower()
    topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
                topics.append(topic)
                break

    if not topics:
        topics.append("other")

    return topics


# ============================================================
# EMOTIONAL STYLE
# ============================================================

EMOTIONAL_WORDS = {
    "positive": [
        "люблю",
        "красиво",
        "красивый",
        "красивая",
        "счастлив",
        "счастье",
        "рад",
        "радует",
        "cute",
        "love",
        "happy",
        "good",
        "great",
    ],

    "negative": [
        "ненавижу",
        "бесит",
        "бесит",
        "плохо",
        "грустно",
        "страшно",
        "устал",
        "устала",
        "проблема",
        "hate",
        "sad",
        "bad",
        "worst",
    ],
}


def detect_emotional_indicators(text):
    lowered = text.lower()

    positive = sum(
        1
        for word in EMOTIONAL_WORDS["positive"]
        if re.search(r"\b" + re.escape(word) + r"\b", lowered)
    )

    negative = sum(
        1
        for word in EMOTIONAL_WORDS["negative"]
        if re.search(r"\b" + re.escape(word) + r"\b", lowered)
    )

    return {
        "positive_indicators": positive,
        "negative_indicators": negative,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    posts = data["posts"]

    if not posts:
        print("No posts found.")
        return

    lengths_chars = []
    lengths_words = []

    languages = Counter()
    formats = Counter()
    question_types = Counter()
    topics = Counter()
    sources = Counter()

    emoji_post_count = 0
    total_emojis = 0

    punctuation_counts = Counter()

    casual_word_counts = Counter()

    first_person_count = 0
    second_person_count = 0

    positive_posts = 0
    negative_posts = 0

    mixing_counts = Counter()

    for post in posts:

        text = post["text"]

        lengths_chars.append(len(text))
        lengths_words.append(len(text.split()))

        languages[detect_language(text)] += 1
        formats[detect_format(text)] += 1
        sources[post.get("source", "unknown")] += 1

        # Questions
        q_type = question_type(text)

        if q_type != "not_question":
            question_types[q_type] += 1

        # Topics
        for topic in detect_topics(text):
            topics[topic] += 1

        # Emojis
        emojis = extract_emojis(text)

        if emojis:
            emoji_post_count += 1
            total_emojis += len(emojis)

        # Punctuation
        punctuation = punctuation_features(text)

        for feature, present in punctuation.items():
            if present:
                punctuation_counts[feature] += 1

        # Casual language
        casual_words = count_casual_indicators(text)

        for word in casual_words:
            casual_word_counts[word] += 1

        # Personal language
        personal = detect_personal_language(text)

        if personal["first_person"]:
            first_person_count += 1

        if personal["second_person"]:
            second_person_count += 1

        # Language mixing
        mixing = detect_mixing(text)

        for language, present in mixing.items():
            if present:
                mixing_counts[language] += 1

        # Emotional indicators
        emotions = detect_emotional_indicators(text)

        if emotions["positive_indicators"] > 0:
            positive_posts += 1

        if emotions["negative_indicators"] > 0:
            negative_posts += 1

    # ========================================================
    # PROFILE
    # ========================================================

    total = len(posts)

    profile = {
        "dataset": {
            "total_posts": total,
            "sources": dict(sources),
        },

        "length": {
            "average_characters": round(mean(lengths_chars), 1),
            "median_characters": round(median(lengths_chars), 1),
            "shortest_characters": min(lengths_chars),
            "longest_characters": max(lengths_chars),
            "average_words": round(mean(lengths_words), 1),
            "median_words": round(median(lengths_words), 1),
        },

        "language": {
            "counts": dict(languages),
            "percentages": {
                language: round(count / total * 100, 1)
                for language, count in languages.items()
            },
        },

        "format": {
            "counts": dict(formats),
            "percentages": {
                format_name: round(count / total * 100, 1)
                for format_name, count in formats.items()
            },
        },

        "questions": {
            "question_posts": sum(question_types.values()),
            "question_percentage": round(
                sum(question_types.values()) / total * 100,
                1
            ),
            "types": dict(question_types),
        },

        "emoji": {
            "posts_with_emoji": emoji_post_count,
            "percentage_with_emoji": round(
                emoji_post_count / total * 100,
                1
            ),
            "total_emojis": total_emojis,
            "average_emojis_per_post": round(
                total_emojis / total,
                2
            ),
        },

        "punctuation": {
            "counts": dict(punctuation_counts),
            "percentages": {
                feature: round(count / total * 100, 1)
                for feature, count in punctuation_counts.items()
            },
        },

        "casual_language": {
            "posts_with_casual_words": sum(
                1
                for post in posts
                if count_casual_indicators(post["text"])
            ),
            "common_words": dict(casual_word_counts.most_common(20)),
        },

        "personal_language": {
            "first_person_posts": first_person_count,
            "first_person_percentage": round(
                first_person_count / total * 100,
                1
            ),
            "second_person_posts": second_person_count,
            "second_person_percentage": round(
                second_person_count / total * 100,
                1
            ),
        },

        "language_mixing": {
            "counts": dict(mixing_counts),
            "percentages": {
                language: round(count / total * 100, 1)
                for language, count in mixing_counts.items()
            },
        },

        "topics": {
            "counts": dict(topics.most_common()),
            "percentages": {
                topic: round(count / total * 100, 1)
                for topic, count in topics.most_common()
            },
        },

        "emotional_style": {
            "posts_with_positive_indicators": positive_posts,
            "posts_with_negative_indicators": negative_posts,
            "positive_percentage": round(
                positive_posts / total * 100,
                1
            ),
            "negative_percentage": round(
                negative_posts / total * 100,
                1
            ),
        },
    }

    # ========================================================
    # SAVE
    # ========================================================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            profile,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # TERMINAL REPORT
    # ========================================================

    print("=" * 60)
    print("UNITEe STYLE ANALYSIS")
    print("=" * 60)

    print(f"\nPosts analyzed: {total}")

    print("\nLength:")
    print(f"  Average characters: {profile['length']['average_characters']}")
    print(f"  Median characters: {profile['length']['median_characters']}")
    print(f"  Average words: {profile['length']['average_words']}")
    print(f"  Median words: {profile['length']['median_words']}")

    print("\nLanguage:")
    for language, count in languages.most_common():
        percentage = count / total * 100
        print(f"  {language}: {count} ({percentage:.1f}%)")

    print("\nFormat:")
    for format_name, count in formats.most_common():
        percentage = count / total * 100
        print(f"  {format_name}: {count} ({percentage:.1f}%)")

    print("\nQuestions:")
    print(
        f"  {sum(question_types.values())} "
        f"({sum(question_types.values()) / total * 100:.1f}%)"
    )

    print("\nQuestion types:")
    for q_type, count in question_types.most_common():
        print(f"  {q_type}: {count}")

    print("\nEmoji:")
    print(
        f"  Posts with emoji: {emoji_post_count} "
        f"({emoji_post_count / total * 100:.1f}%)"
    )
    print(f"  Total emojis: {total_emojis}")
    print(
        f"  Average per post: "
        f"{total_emojis / total:.2f}"
    )

    print("\nCasual language:")
    print(
        f"  Posts containing casual indicators: "
        f"{profile['casual_language']['posts_with_casual_words']}"
    )

    for word, count in casual_word_counts.most_common(15):
        print(f"  {word}: {count}")

    print("\nPersonal language:")
    print(
        f"  First person: {first_person_count} "
        f"({first_person_count / total * 100:.1f}%)"
    )
    print(
        f"  Second person: {second_person_count} "
        f"({second_person_count / total * 100:.1f}%)"
    )

    print("\nTopics:")
    for topic, count in topics.most_common():
        print(
            f"  {topic}: {count} "
            f"({count / total * 100:.1f}%)"
        )

    print("\nEmotional indicators:")
    print(
        f"  Positive: {positive_posts} "
        f"({positive_posts / total * 100:.1f}%)"
    )
    print(
        f"  Negative: {negative_posts} "
        f"({negative_posts / total * 100:.1f}%)"
    )

    print(f"\nSaved profile to: {OUTPUT_FILE}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

