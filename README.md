REQUIREMENTS
------------
  pip install MetaTrader5 pandas numpy

USAGE
-----
  trading_bot.py --symbols XAUUSD+ EURUSD+ --dashboard-port 9000

DASHBOARD
---------
  The bot starts a built-in web server automatically.
  Browser opens at http://localhost:9000/

TELEGRAM ALERTS
---------------
  1. Create a bot via @BotFather on Telegram -> copy the token
  2. Start a chat with your bot, then get your chat_id:
       https://api.telegram.org/bot<TOKEN>/getUpdates
  3. Add to config.json:
       "telegram": {
           "enabled": true,
           "bot_token": "123456:ABC-...",
           "chat_id": "-100123456789",
           "alert_new_grid": true,
           "alert_grid_pnl": true
       }
