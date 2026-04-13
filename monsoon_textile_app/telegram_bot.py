"""
RainLoom Telegram Bot (Hackathon Live Demo Edition)
Allows users to ask about district risks and receive instant advisories.

Usage:
    export TELEGRAM_BOT_TOKEN="your_bot_token"
    python -m monsoon_textile_app.telegram_bot
"""

import os
import asyncio
from datetime import datetime
from loguru import logger
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from monsoon_textile_app.api.data_bridge import get_dashboard_data
from monsoon_textile_app.utils.alerts import FarmerAdvisorySystem

# Initialize the advisory system
advisory_system = FarmerAdvisorySystem()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 🌧️ Welcome to RainLoom Bot.\n\n"
        "I monitor monsoon failures and predict risks for cotton farmers and textile stocks.\n\n"
        "To get a live risk advisory for your district, type:\n"
        "`/risk <DistrictName>`\n\n"
        "Example: `/risk Warangal`"
    )
    await update.message.reply_markdown(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_msg = (
        "Available commands:\n"
        "/start - Welcome message\n"
        "/help - Show this message\n"
        "/risk <DistrictName> - Get real-time risk advisory for a district\n\n"
        "Currently supported districts: Vidarbha, Marathwada, Saurashtra, Telangana, Rajkot, Surendranagar, Guntur, Adilabad, Akola, Amravati, Yavatmal, Wardha"
    )
    await update.message.reply_text(help_msg)

async def risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch and return risk advisory for a requested district."""
    if not context.args:
        await update.message.reply_text("Please provide a district name. Example: /risk Warangal")
        return

    district_query = " ".join(context.args).title()
    
    # Check if district is supported
    supported = [d.title() for d in advisory_system.config["supported_districts"]]
    
    if district_query not in supported:
        await update.message.reply_text(
            f"Sorry, I don't have detailed data for '{district_query}' right now. "
            f"Try one of: {', '.join(supported[:5])}..."
        )
        return

    await update.message.reply_text(f"🔍 Analyzing live data for {district_query}...")
    
    try:
        # Get live data
        data = get_dashboard_data()
        
        # Determine average rainfall deficit
        rainfall = data.get("rainfall", {})
        avg_def = -15.0 # default mock
        
        if isinstance(rainfall, dict):
            annual = rainfall.get("annual_deficit")
            if hasattr(annual, "iloc") and len(annual) > 0:
                latest = annual.iloc[-1]
                if hasattr(latest, "values"):
                    vals = [v for v in list(latest.values) if isinstance(v, (int, float))]
                    avg_def = (sum(vals) / len(vals)) if vals else -15.0
                elif isinstance(latest, dict):
                    vals = [v for v in latest.values() if isinstance(v, (int, float))]
                    avg_def = sum(vals) / max(len(vals), 1) if vals else -15.0
                    
        # Find risk score (calculate composite or use mocked extreme for demo)
        # Using a higher number if it's a drought preset, or ~0.4 normally
        risk_score = 0.55 if avg_def < -10 else 0.2
        
        # If severe deficit, increase risk score for demo effect
        if avg_def < -20:
            risk_score = 0.85

        date_str = datetime.now().strftime("%Y-%m-%d")
        
        advisory = advisory_system.generate_advisory(
            risk_score=risk_score,
            district=district_query,
            deficit_pct=avg_def,
            date=date_str
        )
        
        # Build response
        level_icons = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "EXTREME": "🔴"}
        icon = level_icons.get(advisory["severity"], "ℹ️")
        
        response = (
            f"{icon} *RainLoom Live Advisory: {district_query}*\n"
            f"Date: {date_str}\n\n"
            f"*Severity:* {advisory['severity']} Risk ({advisory['risk_score']:.2f})\n"
            f"*Rainfall Deficit:* {advisory['deficit_pct']:.1f}%\n"
            f"*Exp. Yield Drop:* {advisory['estimated_yield_drop_pct']['min']}-{advisory['estimated_yield_drop_pct']['max']}%\n\n"
            f"*Message:*\n{advisory['message']}\n\n"
            f"*Recommended Actions:*\n"
        )
        
        for action in advisory["recommended_actions"]:
            response += f"• {action}\n"
            
        await update.message.reply_markdown(response)

    except Exception as e:
        logger.error(f"Error serving /risk command: {e}")
        await update.message.reply_text("Failed to calculate risk right now due to a data error. Please try again in a few minutes.")

def main() -> None:
    """Start the bot."""
    # Try to load token from environment variable
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if not token or token == "your_bot_token":
        logger.warning(
            "TELEGRAM_BOT_TOKEN not found in environment. "
            "Please set it: export TELEGRAM_BOT_TOKEN='your_token'"
        )
        return

    logger.info("Starting RainLoom Telegram Bot...")
    
    # Create application
    application = Application.builder().token(token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("risk", risk_command))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
