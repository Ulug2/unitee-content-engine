import json
import re
from collections import Counter
from statistics import mean

INPUT_FILE = "raw_telegram_posts.json"


def detect_language(text):
    has_cyrillic = bool(re.search(r"[а-яәіңғқөұүһі]", text.lower()))
    has_latin = bool(re.search(r"[a-z]", text.lower()))

    if has_cyrillic and has_latin:
        return "mixed"
    elif has_cyrillic:
        return "cyrillic"
    elif has_latin:
        return "english_or_latin"
    else:
        return "other"


def classify_format(text):
    stripped = text.strip()

    if "?" in stripped:
        return "question"
    if re.search(r"^(1|2|3|а|б)[.)-]", stripped.lower()):
        return "poll_or_options"
    if len(stripped.split()) <= 8:
        return "short_statement"
    if len(stripped.split()) >= 80:
        return "long_story"
    return "general_statement"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    posts = data["posts"]

    lengths = [len(post["text"]) for post in posts]
    languages = Counter(detect_language(post["text"]) for post in posts)
    formats = Counter(classify_format(post["text"]) for post in posts)
    sources = Counter(post["source"] for post in posts)

    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal posts: {len(posts)}")
    print(f"Average characters per post: {mean(lengths):.1f}")
    print(f"Shortest post: {min(lengths)} characters")
    print(f"Longest post: {max(lengths)} characters")

    print("\nPosts by source:")
    for source, count in sources.items():
        print(f"  {source}: {count}")

    print("\nLanguage indicators:")
    for language, count in languages.most_common():
        print(f"  {language}: {count}")

    print("\nPost formats:")
    for post_format, count in formats.most_common():
        print(f"  {post_format}: {count}")

    print("\nSample posts:")
    for index, post in enumerate(posts[:10], start=1):
        print(f"\n--- Sample {index} ({post['source']}) ---")
        print(post["text"][:500])


if __name__ == "__main__":
    main()
