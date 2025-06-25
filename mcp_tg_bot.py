import logging
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from mcp_bot import collection_mcp, mcp_instance, answer_question_with_mcp_rules

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN not found in .env file or environment variables.")
    exit()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message on /start."""
    await update.message.reply_text(
        "Hi! I am your HODLToken AI Assistant, powered by the official hodltoken.net website. How can I help you today?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message on /help."""
    help_text = (
        "I can help you with information about HODLToken, including:\n"
        "➡️ $HODL Token details & Tokenomics\n"
        "➡️ BNB rewards system\n"
        "➡️ HODL NFTs (HODL Hands®, Gem Fighter NFTs)\n"
        "➡️ Play-to-Earn games (Crypto Slash, Gem Miner, etc.)\n"
        "➡️ The HODL App features and guides\n"
        "➡️ Project roadmap, whitepaper, and security.\n\n"
        "Just ask your question! For example: 'Tell me about Gem Fighter NFTs' or 'How does HODL staking work?'"
    )
    await update.message.reply_text(help_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user text messages and respond with answers from MCP model."""
    user_query = update.message.text.strip()
    logger.info(f"User query: {user_query}")

    normalized_query = user_query.lower()
    if normalized_query in ["what are you?", "who are you?"]:
        await update.message.reply_text(
            "I'm the HODLToken AI Assistant! I provide information based on the official hodltoken.net website."
        )
        return
    if normalized_query in ["who made you?", "who created you?"]:
        await update.message.reply_text(
            "I'm an AI assistant developed to help with your HODLToken questions, using data from the official website and powered by advanced language models."
        )
        return

    greetings = [
        "hi",
        "hello",
        "hey",
        "yo",
        "what's up",
        "good morning",
        "good afternoon",
        "good evening",
        "sup",
    ]
    greeting_starters = ["hi ", "hello ", "hey "]
    if normalized_query in greetings or any(
        normalized_query.startswith(g) for g in greeting_starters
    ):
        await update.message.reply_text(
            "Hey there! 👋 How can I assist you with HODLToken today?"
        )
        return

    await update.message.reply_text(f"Thinking about '{user_query}'...")

    try:
        llm_answer = answer_question_with_mcp_rules(user_query)
        if len(llm_answer) > 4050:  # Telegram message length limit margin
            llm_answer = llm_answer[:4000] + "\n\n[...Answer truncated due to length...]"
        await update.message.reply_text(llm_answer, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        await update.message.reply_text(
            "I'm sorry, I encountered an issue while trying to process your request. Please try again."
        )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors."""
    logger.error(f"Update {update} caused error {context.error}")


def main():
    if not collection_mcp or not mcp_instance or not mcp_instance.generator:
        logger.error(
            "Critical components (DB or LLM via MCPModel) from mcp_bot are not initialized. Telegram bot may not function correctly."
        )

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    )
    application.add_error_handler(error_handler)

    logger.info("Starting HODLToken Telegram Assistant Bot...")
    application.run_polling()


if __name__ == "__main__":
    main()
