import asyncio
from telethon import TelegramClient
from dotenv import load_dotenv
import os

load_dotenv()

API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")

client = TelegramClient("unitee_session", API_ID, API_HASH)


async def inspect_channel(username, name):
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)

    entity = await client.get_entity(username)

    print(f"Title: {entity.title}")
    print(f"Username: @{entity.username}")
    print(f"Channel ID: {entity.id}")

    print("\nLatest 20 messages:\n")

    count = 0

    async for message in client.iter_messages(entity, limit=20):
        if not message.text:
            continue

        count += 1

        print(f"--- Message {count} ---")
        print(f"ID: {message.id}")
        print(f"Date: {message.date}")
        print(f"Views: {message.views}")
        print(f"Text:\n{message.text[:1000]}")
        print()


async def main():
    await client.start()

    await inspect_channel(
        "https://t.me/nutumba",
        "TUMBA"
    )

    await inspect_channel(
        "https://t.me/sdu_angme",
        "SDU ANGME"
    )

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
