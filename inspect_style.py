import json
import random
from collections import Counter


INPUT_FILE = "safe_style_posts.json"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    posts = data["posts"]

    print("=" * 70)
    print("RANDOM STYLE SAMPLE")
    print("=" * 70)

    sample_size = min(40, len(posts))
    sample = random.sample(posts, sample_size)

    for i, post in enumerate(sample, 1):
        print(f"\n{i}.")
        print(post["text"])

    print("\n" + "=" * 70)
    print("LENGTH DISTRIBUTION")
    print("=" * 70)

    buckets = Counter()

    for post in posts:
        length = len(post["text"])

        if length <= 30:
            buckets["0-30"] += 1
        elif length <= 60:
            buckets["31-60"] += 1
        elif length <= 100:
            buckets["61-100"] += 1
        elif length <= 150:
            buckets["101-150"] += 1
        elif length <= 250:
            buckets["151-250"] += 1
        else:
            buckets["250+"] += 1

    for bucket, count in buckets.items():
        print(f"{bucket}: {count}")

    print("\n" + "=" * 70)
    print("WORD LENGTH DISTRIBUTION")
    print("=" * 70)

    word_buckets = Counter()

    for post in posts:
        words = len(post["text"].split())

        if words <= 5:
            word_buckets["0-5"] += 1
        elif words <= 10:
            word_buckets["6-10"] += 1
        elif words <= 20:
            word_buckets["11-20"] += 1
        elif words <= 30:
            word_buckets["21-30"] += 1
        elif words <= 50:
            word_buckets["31-50"] += 1
        else:
            word_buckets["50+"] += 1

    for bucket, count in word_buckets.items():
        print(f"{bucket}: {count}")


if __name__ == "__main__":
    main()
