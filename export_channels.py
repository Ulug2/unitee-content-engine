import asyncio
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()

API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")

client = TelegramClient("unitee_session", API_ID, API_HASH)

# Source channel URLs are read from the environment rather than hardcoded, so the source
# community isn't identifiable from this file alone. Set CHANNEL_A_URL / CHANNEL_B_URL in .env.
CHANNELS = {
    "channel_a": os.getenv("CHANNEL_A_URL"),
    "channel_b": os.getenv("CHANNEL_B_URL"),
}

MESSAGES_PER_CHANNEL = 100
OUTPUT_FILE = "raw_telegram_posts.json"


async def export_channel(channel_key, channel_url):
    print(f"\nExporting: {channel_key}")

    entity = await client.get_entity(channel_url)
    posts = []

    async for message in client.iter_messages(
        entity,
        limit=MESSAGES_PER_CHANNEL,
    ):
        if not message.text or not message.text.strip():
            continue

        posts.append({
            "source": channel_key,
            "telegram_message_id": message.id,
            "date": message.date.isoformat() if message.date else None,
            "text": message.text.strip(),
        })

    print(f"Collected {len(posts)} text posts from {channel_key}")
    return posts


async def main():
    await client.start()

    all_posts = []

    for channel_key, channel_url in CHANNELS.items():
        if not channel_url:
            raise RuntimeError(f"Missing URL for '{channel_key}' - set it in .env (see .env.example).")
        posts = await export_channel(channel_key, channel_url)
        all_posts.extend(posts)

    output = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_posts": len(all_posts),
        "posts": all_posts,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\nExport complete!")
    print(f"Total posts: {len(all_posts)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())
