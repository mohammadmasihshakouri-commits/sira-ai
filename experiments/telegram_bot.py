import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram import Update


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import app
from core.channel_adapters import telegram_adapter
from core.sira_runtime import handle_message
from core.business_profile_loader import get_resolved_business_agent
from core.response_variants import random_greeting

load_dotenv(PROJECT_ROOT / ".env")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

business_agent = get_resolved_business_agent("cinematicket", "support")
enabled_playbooks = business_agent.get("enabled_playbooks", [])

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "درخواست شما برای اپراتور ارسال شد.\n"
        "تا چند ثانیه دیگر پاسخگو خواهد بود..."
    )

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    import asyncio
    await asyncio.sleep(4)

    greeting = random_greeting()

    agent_identity = business_agent.get("agent_identity", {})
    agent_name = agent_identity.get("agent_name", "دستیار هوشمند")
    business_name = business_agent.get("business_name", "این مجموعه")

    await update.message.reply_text(
        f"{greeting}\n"
        f"من {agent_name} هستم، پشتیبان {business_name} 🌸\n"
        "بفرمایید چطور می‌تونم راهنمایی‌تون کنم؟"
    )

async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat = update.effective_chat
    text = update.message.text
    print("USER ID:", user.id)
    raw_update = {
        "user_id": user.id,
        "chat_id": chat.id,
        "username": user.username,
        "text": text,
    }

    channel_message = telegram_adapter(raw_update)

    result = handle_message(
        channel_message=channel_message,
        conversation_state=None,
        enabled_playbooks=enabled_playbooks,
    )

    await update.message.reply_text(result["reply"])


def main():
    if not TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))

    print("Sira Telegram bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
