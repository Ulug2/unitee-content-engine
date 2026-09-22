import os
import asyncio
import subprocess

import qrcode
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

load_dotenv()

API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")

client = TelegramClient("unitee_session", API_ID, API_HASH)


async def main():
    await client.connect()

    if not await client.is_user_authorized():
        print("\nGenerating Telegram login QR code...")

        qr_login = await client.qr_login()

        qr = qrcode.make(qr_login.url)
        qr_path = os.path.abspath("telegram_login_qr.png")
        qr.save(qr_path)

        print(f"\nQR code saved to:")
        print(qr_path)
        print("\nOpening QR code...")

        subprocess.run(["open", qr_path])

        print("\n==========================================")
        print("SCAN THE QR CODE WITH YOUR TELEGRAM APP")
        print("==========================================")
        print("\nOn your phone:")
        print("Telegram → Settings → Devices → Link Desktop Device")
        print("Then scan the QR code on your Mac.")
        print("\nWaiting for approval...")

        try:
            await qr_login.wait()
        except SessionPasswordNeededError:
            print("\nYour Telegram account has 2FA enabled.")
            password = input("Enter your Telegram 2FA password: ")
            await client.sign_in(password=password)

        print("\nSuccessfully logged into Telegram!")

    me = await client.get_me()

    print("\nLogged in as:")
    print(f"Name: {me.first_name}")
    print(f"Username: @{me.username}" if me.username else "Username: none")

    print("\nChecking Tumba...")
    tumba = await client.get_entity("https://t.me/nutumba")
    print(f"Tumba: {tumba.title}")

    print("\nChecking SDU Angme...")
    sdu = await client.get_entity("https://t.me/sdu_angme")
    print(f"SDU Angme: {sdu.title}")

    print("\nConnection test successful!")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())