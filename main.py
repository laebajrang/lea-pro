from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import os
from signal_core import LAESignalCore
from telegram import Bot, TelegramError
from datetime import datetime
import pytz
import json
import logging
from pathlib import Path
from fastapi.security import APIKeyHeader

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(_name_)

app = FastAPI(
    title="LAE Pro Trading Signal API",
    description="Automated trading signal generation and distribution system",
    version="2.1"
)

# Security
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Configuration
class Config:
    def _init_(self):
        self.timezone = pytz.timezone("Asia/Kolkata")
        self.signal_log_file = Path("data/lae_signal_log.json")
        self.signal_log_file.parent.mkdir(exist_ok=True)
        
        # Telegram settings - Load from environment with fallback
        self.bot_token = os.getenv("BOT_TOKEN", "7972168108:AAEPQy41g-KgONWT5JURLPelPYhk41HWMx0")
        self.chat_id = os.getenv("CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.error("Missing required Telegram configuration")
            raise ValueError("Telegram credentials not configured")

config = Config()

# Initialize components
signal_engine = LAESignalCore()
bot = Bot(token=config.bot_token)

# Models
class MarketInput(BaseModel):
    market_data: Dict
    news_data: Dict
    api_key: Optional[str] = None

class SignalResponse(BaseModel):
    status: str
    data: Optional[Dict] = None
    message: Optional[str] = None

# Helper functions
def save_signal(signal: Dict) -> bool:
    """Save signal to JSON log file with rotation"""
    try:
        # Load existing logs
        if config.signal_log_file.exists():
            with open(config.signal_log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new signal
        signal["timestamp"] = datetime.now(config.timezone).strftime("%Y-%m-%d %H:%M:%S")
        logs.append(signal)
        
        # Save with pretty print
        with open(config.signal_log_file, "w") as f:
            json.dump(logs, f, indent=2)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to save signal: {str(e)}")
        return False

def format_telegram_message(signal: Dict) -> str:
    """Create formatted Telegram message"""
    message = [
        f"📡 LAE Pro Trade Signal",
        f"🕒 Time: {signal['timestamp']}",
        f"📊 Decision: {signal['decision']}",
        f"🤖 Confidence: {signal['confidence']}%",
        f"🧠 Logic: {', '.join(signal.get('logic_used', [])) or 'None'}"
    ]
    
    if signal['decision'] != "NO TRADE":
        message.extend([
            f"\n💰 Entry: {signal.get('entry')}",
            f"🛡 SL: {signal.get('stop_loss')}",
            f"🎯 TP: {signal.get('take_profit')}",
            f"⚖ RRR: {signal.get('risk_reward_ratio', 'N/A')}"
        ])
    
    if signal.get('reason'):
        message.append(f"\n⚠ Reason: {signal['reason']}")
    
    return "\n".join(message)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """API key verification middleware"""
    expected_key = os.getenv("API_SECRET_KEY")
    if not expected_key or api_key == expected_key:
        return True
    raise HTTPException(status_code=403, detail="Invalid API Key")

# Endpoints
@app.post("/send-signal", response_model=SignalResponse)
async def send_signal(
    input_data: MarketInput,
    authorized: bool = Depends(verify_api_key)
):
    """Generate and distribute trading signal"""
    try:
        # Generate signal
        result = signal_engine.generate_signal(
            market_data=input_data.market_data,
            news_data=input_data.news_data
        )
        
        # Save to log
        if not save_signal(result):
            raise HTTPException(status_code=500, detail="Failed to save signal")
        
        # Send Telegram notification
        try:
            msg = format_telegram_message(result)
            await bot.send_message(
                chat_id=config.chat_id,
                text=msg,
                parse_mode="Markdown"
            )
        except TelegramError as e:
            logger.error(f"Telegram send failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Signal generated but Telegram notification failed"
            )
        
        return {
            "status": "success",
            "data": result
        }
    
    except Exception as e:
        logger.error(f"Signal generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Signal generation failed")

@app.get("/signal-log", response_model=SignalResponse)
async def get_signal_log(authorized: bool = Depends(verify_api_key)):
    """Retrieve signal log history"""
    try:
        if not config.signal_log_file.exists():
            return {
                "status": "success",
                "data": [],
                "message": "No signal log found"
            }
        
        with open(config.signal_log_file, "r") as f:
            logs = json.load(f)
        
        return {
            "status": "success",
            "data": logs
        }
    
    except Exception as e:
        logger.error(f"Log retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")

@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(config.timezone).strftime("%Y-%m-%d %H:%M:%S"),
        "version": app.version
    }

if _name_ == "_main_":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE")
    )
