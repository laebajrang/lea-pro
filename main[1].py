import os
from dotenv import load_dotenv
from telegram.ext import Updater, CommandHandler

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

def start(update, context):
    update.message.reply_text("🚀 LAE Pro Activated!\nReal-time signals incoming...")

def signal(update, context):
    update.message.reply_text("📈 BTCUSDT Buy @ $108800 | SL: $107200 | Target: $111500")

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("signal", signal))

updater.start_polling()