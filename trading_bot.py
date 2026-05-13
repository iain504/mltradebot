# trading_bot_v7_0_8_prod_v14.py
# Multi-symbol concurrent trading bot for MetaTrader 5
# Version: v14 -- Win-rate improvements pass
#
# WIN-RATE IMPROVEMENTS (v14):
# [P1] Direction classification instead of raw price regression
#      LSTM + XGBoost now predict up/flat/down over next N bars (3-class).
#      Removes the impossible task of exact price prediction.
# [P2] Removed +/-3% prediction clamp that suppressed breakout signals.
#      Replaced with ATR-normalised validation bounds.
# [P3] Learned ensemble weights via recent validation error (stacking).
#      Weights updated every prediction; no longer fixed guesses.
# [C1] RMSE confidence filter now ATR-normalised (rmse / atr).
#      Previously raw RMSE=0.3 was meaningless across multi-symbol bot.
# [C2] Removed confidence_factor floor of 0.5 that prevented blocking trades.
# [E1] Trend filter gate: buys only when EMA20 > SMA50; sells opposite.
# [E2] ADX gate: entries blocked when ADX < 20 (ranging/choppy market).
# [E3] Q-table state space reduced: 3 features x 5 bins (was 6 x 10).
#      Prevents near-total sparsity that made RL degenerate.
# [E4] Trade cooldown now ATR-velocity-based, not hard 3600 s clock.
# [S1] SL multiplier now symbol_type-aware (forex 1.5x, stock 2.0x ATR).
# [S2] Exit signal uses ATR-normalised return threshold (not +/-3% of price).
# [S3] Max-bars-open time stop added; stale positions closed after N bars.
# [T1] Walk-forward cross-validation replaces contiguous 80/20 split.
#      Prevents temporal look-ahead leakage in validation set.
# [T2] Session time features added: hour_sin, hour_cos, dow_sin, dow_cos.
# [T3] VWAP resets per session (daily); rolling 20-bar VWAP as backup.
# [T4] Feature selection: bottom 30% by XGBoost importance pruned.

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator, CCIIndicator
from ta.volatility import BollingerBands
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import logging
import asyncio
import json
import os
import argparse
import platform
import warnings
import hashlib
import time
import pickle
import threading
import concurrent.futures
from collections import deque
import math
import signal
import sys
import smtplib
from email.mime.text import MIMEText
from sklearn.preprocessing import KBinsDiscretizer
# keras_tuner and matplotlib are lazy-imported only where needed to avoid
# adding ~200 MB of startup cost when use_tuner=False (the default).

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

# ---------------------------------------------------------------------------
# Global Locks and Caches (Phase 1,3)
# ---------------------------------------------------------------------------
mt5_lock = threading.Lock()  # threading.Lock so it works across per-symbol threads
data_cache = {}              # Symbol -> DataFrame cache for appending new bars

# Async context manager wrapper so all existing `async with mt5_lock:` calls still work
class _AsyncMT5Lock:
    """Wraps a threading.Lock as an async context manager."""
    def __init__(self):
        self._lock = threading.Lock()
    def __enter__(self):  return self._lock.__enter__()
    def __exit__(self, *a): return self._lock.__exit__(*a)
    async def __aenter__(self): self._lock.acquire(); return self
    async def __aexit__(self, *a): self._lock.release()

mt5_lock = _AsyncMT5Lock()

# ---------------------------------------------------------------------------
# Dashboard state -- written by each bot thread, read by the dashboard thread
# ---------------------------------------------------------------------------
_dashboard_state: dict = {}          # symbol -> dict of status fields
_dashboard_lock  = threading.Lock()  # protects _dashboard_state writes

_pnl_history: dict = {}   # symbol -> deque of (timestamp, pnl_float) for sparklines

def update_dashboard(symbol: str, **kwargs):
    """Thread-safe update of a symbol's dashboard row.
    Full implementation is defined later in the file after run_dashboard;
    this forward declaration is overwritten at module load time."""
    with _dashboard_lock:
        if symbol not in _dashboard_state:
            _dashboard_state[symbol] = {
                'status':       'starting',
                'price':        '-',
                'position':     '-',
                'pnl':          '-',
                'confidence':   '-',
                'action':       '-',
                'bars':          0,
                'trades_today':  0,
                'last_update':  '-',
                'error':        '',
                'sl':           '-',
                'entry':        '-',
            }
        _dashboard_state[symbol].update(kwargs)
        _dashboard_state[symbol]['last_update'] = datetime.now().strftime('%H:%M:%S')
        # Track P&L history for sparkline
        pnl_raw = kwargs.get('pnl', '')
        try:
            pnl_val = float(str(pnl_raw).replace('+', ''))
            if symbol not in _pnl_history:
                _pnl_history[symbol] = deque(maxlen=50)
            _pnl_history[symbol].append((time.time(), pnl_val))
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Logger:
    _configured = False  # Class-level flag so basicConfig only runs once

    def __init__(self, log_file="trading_bot.log"):
        if not Logger._configured:
            import io, sys as _sys
            # File handler: always UTF-8
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            # Stream handler: wrap stdout in UTF-8 with 'replace' so Windows cp1252
            # console never raises UnicodeEncodeError on symbols like <=, >=, ->, x
            try:
                utf8_stdout = io.TextIOWrapper(
                    _sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
                )
                stream_handler = logging.StreamHandler(stream=utf8_stdout)
            except AttributeError:
                # stdout has no .buffer (e.g. already wrapped) -- fall back to default
                stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root = logging.getLogger()
            root.setLevel(logging.INFO)
            root.addHandler(file_handler)
            root.addHandler(stream_handler)
            Logger._configured = True
        self.logger = logging.getLogger("trading_bot")

    def info(self, message): self.logger.info(message)
    def warning(self, message): self.logger.warning(message)
    def error(self, message): self.logger.error(message)
    def debug(self, message): self.logger.debug(message)

# ---------------------------------------------------------------------------
# Configuration (Item 11: Symbol-specific configs)
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, config_file="config.json"):
        self.default_config = {
            "timeframe": "TIMEFRAME_M15",
            "max_retries": 5,
            "retry_delay": 5,
            "lookback": 30,
            "warmup_period": 20,
            "epsilon": 0.1,
            "epsilon_decay": 0.996,
            "epsilon_min": 0.01,
            "alpha": 0.08,
            "gamma": 0.92,
            "risk": {
                "risk_percent":           1.5,
                "daily_loss_limit":       5.0,
                "max_drawdown":           15.0,
                "daily_profit_target":    0,
                "equity_profit_threshold": 0
            },
            "retrain_interval": 100,
            "wf_splits": 5,
            "model_max_age_hours": 24,   # Reload cached models if younger than this; retrain if older
            "initial_capital": 10000,
            "time_zone_offset_hours": -5,
            "use_tuner": False,
            "tune_on_drift": False,
            "drift_acc_threshold": 0.40,
            "drift_min_samples":      50,
            "drift_retrain_cooldown": 50,
            "min_confidence": 0.45,
            "max_bars_open": 12,
            "predict_horizon": 6,          # align to typical hold time; override per symbol
            "flat_thresh_mult": 0.3,       # ATR fraction for flat zone; 0.2 forex, 0.5 crypto
            "num_symbols": 1,
            "breakeven_atr_trigger":  1.0,
            "trailing_atr_multiplier": {
                "forex":  1.0,
                "stock":  1.2,
                "crypto": 1.5
            },
            "use_q_table":            False,
            "adx_min_entry":          18,     # lowered from 20 -- 20 blocked too many valid setups
            "spread_max_multiplier":  2.5,
            "session_gate":           True,
            "session_block_start":    0,      # UTC hour: start of dead-of-night block (default 00:00)
            "session_block_end":      5,      # UTC hour: end of dead-of-night block (default 05:00)
            "mtf_confirmation":       True,
            "correlation_guard":      True,
            "cooldown_min_bars":      1,      # minimum bars between trades (was hardcoded to 3)
            "log_filter_blocks":      True,
            "confidence_override_threshold": 0.90,
            "trend_filter_enabled":   True,
            "adx_filter_enabled":     True,
            "symbol_watch_interval":  10,    # seconds between symbols.json checks for hot add/remove
            "forex_close_fri_utc":    21,
            "forex_open_sun_utc":     21,
            "telegram_token":         "",
            "telegram_chat_id":       "",
            "sl_atr_multiplier": {
                "forex":  1.5,
                "stock":  2.0,
                "crypto": 2.5
            },
            "partial_tp": {
                "enabled": True,
                "percent": 0.5,
                "rr": 1.0
            },
            "take_profit": {
                "enabled": True,
                "base_multiplier": 2.0,
                "max_multiplier": 4.0,
                "scale_with_confidence": True,
                "scale_with_volatility": True
            },
            "smtp_host": "",
            "smtp_port": 587,
            "smtp_from": "",
            "smtp_to": "",
            "smtp_user": "",
            "smtp_password": ""
        }
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {"default": self.default_config}
        except Exception as e:
            raise e
        # Ensure "default" exists
        if "default" not in self.config:
            self.config["default"] = self.default_config

    def get(self, key, default=None, symbol=None):
        if symbol and symbol in self.config:
            return self.config[symbol].get(
                key,
                self.config["default"].get(key, self.default_config.get(key, default))
            )
        return self.config["default"].get(key, self.default_config.get(key, default))

# ---------------------------------------------------------------------------
# DataFetcher (Item 8: Dynamic GMT offset, Item 13: Cache)
# ---------------------------------------------------------------------------
class DataFetcher:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        tf_name = config.get('timeframe', 'TIMEFRAME_M15')
        self.timeframe = getattr(mt5, tf_name, mt5.TIMEFRAME_M15)
        self.max_retries = config.get('max_retries', 5)
        self.retry_delay = config.get('retry_delay', 5)
        # Rolling average spread (ask-bid in points) per symbol -- used by spread filter
        self._avg_spread: dict = {}

    def initialize_mt5(self):
        for attempt in range(self.max_retries):
            try:
                if mt5.initialize():
                    self.logger.info("MT5 initialized successfully")
                    return True
            except Exception as e:
                self.logger.warning(f"MT5 initialize exception: {e}")
            self.logger.warning(f"MT5 initialization failed, attempt {attempt + 1}/{self.max_retries}")
            time.sleep(self.retry_delay)
        self.logger.error("MT5 initialization failed after all retries")
        return False

    async def fetch_data(self, symbol, bars=5000):
        global data_cache
        min_bars = self.config.get('lookback', 30) + 50
        cache_max_bars = max(bars, 6000)  # Cap cache to avoid unbounded RAM growth
        if symbol in data_cache and len(data_cache[symbol]) >= min_bars:
            last_time = data_cache[symbol].index[-1].timestamp()
            async with mt5_lock:
                new_rates = mt5.copy_rates_range(
                    symbol, self.timeframe,
                    datetime.fromtimestamp(last_time + 1, tz=timezone.utc),
                    datetime.now(timezone.utc)
                )
            if new_rates is not None and len(new_rates) > 0:
                new_df = pd.DataFrame(new_rates)
                # Fix: utc=True so index is tz-aware -- matches the main cache index
                new_df['time'] = pd.to_datetime(new_df['time'], unit='s', utc=True)
                new_df.set_index('time', inplace=True)
                combined = pd.concat([data_cache[symbol], new_df]).drop_duplicates()
                # Rolling cap: keep only the most recent cache_max_bars rows
                data_cache[symbol] = combined.iloc[-cache_max_bars:]
                self.logger.info(f"Appended {len(new_rates)} new bars to cache for {symbol} (total={len(data_cache[symbol])})")
            return data_cache[symbol].tail(bars)
        # Full fetch if no cache or insufficient
        # Derive bar duration once, outside the retry loop, so it's always in scope
        _tf_minutes_map = {
            mt5.TIMEFRAME_M1: 1, mt5.TIMEFRAME_M5: 5, mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30, mt5.TIMEFRAME_H1: 60, mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440,
        }
        timeframe_minutes = _tf_minutes_map.get(self.timeframe, 15)

        async with mt5_lock:
            for attempt in range(self.max_retries):
                self.logger.info(f"Fetching {bars} bars for {symbol}, attempt {attempt + 1}/{self.max_retries}")
                try:
                    rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, bars)
                except Exception as e:
                    self.logger.warning(f"mt5.copy_rates_from_pos exception: {e}")
                    rates = None

                if rates is None or len(rates) == 0:
                    self.logger.warning(f"No data fetched on direct call, trying range-based fetch")
                    end_time = datetime.now(timezone.utc)
                    rates = mt5.copy_rates_range(symbol, self.timeframe, end_time - timedelta(minutes=timeframe_minutes * bars), end_time)

                if rates is not None and len(rates) >= min_bars:
                    df = pd.DataFrame(rates)
                    # Parse timestamps as UTC-aware so comparisons with datetime.now(timezone.utc) never raise
                    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                    df.set_index('time', inplace=True)
                    # Check for gaps/duplicates
                    if len(df) != len(df.drop_duplicates(subset=['open', 'high', 'low', 'close'])):
                        self.logger.warning(f"Duplicates detected in data for {symbol}, dropping")
                        df = df.drop_duplicates()
                    time_diffs = df.index.to_series().diff().dropna().dt.total_seconds()
                    expected_diff = timeframe_minutes * 60  # Derived from actual timeframe
                    if (time_diffs > expected_diff * 2).any():  # Gap > 2 bars
                        self.logger.warning(f"Gaps detected in data for {symbol}")
                    data_cache[symbol] = df  # Cache full data
                    required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
                    df = df.copy()
                    if not all(col in df.columns for col in required_columns):
                        self.logger.error(f"Missing required columns in data for {symbol}: {df.columns}")
                        return None
                    if not pd.api.types.is_numeric_dtype(df['close']):
                        self.logger.error(f"Non-numeric 'close' column in data for {symbol}")
                        return None
                    if df['close'].nunique() < 2:
                        self.logger.error(f"Close prices are constant or nearly constant: unique values={df['close'].nunique()}")
                        return None
                    try:
                        df_filtered = df[df['tick_volume'] < df['tick_volume'].quantile(0.99)]
                        if len(df_filtered) < min_bars:
                            self.logger.warning("Volume filter removed too many rows, skipping filter")
                        else:
                            df = df_filtered
                    except Exception as e:
                        self.logger.warning(f"Volume filter issue: {e}")
                    try:
                        df_filtered = df[df['close'].pct_change().abs() < 0.1]
                        if len(df_filtered) < min_bars:
                            self.logger.warning("Price-change filter removed too many rows, skipping filter")
                        else:
                            df = df_filtered
                    except Exception as e:
                        self.logger.warning(f"Price change filter issue: {e}")

                    try:
                        q_low = df['close'].quantile(0.005)
                        q_high = df['close'].quantile(0.995)
                        df_filtered = df[df['close'].between(q_low, q_high)]
                        if len(df_filtered) >= min_bars:
                            df = df_filtered
                    except Exception as e:
                        self.logger.warning(f"Quantile filter issue: {e}")

                    if len(df) >= min_bars:
                        close_stats = df['close'].describe()
                        self.logger.info(f"Filtered data stats for {symbol}: rows={len(df)}, min={close_stats.get('min', np.nan):.2f}, max={close_stats.get('max', np.nan):.2f}, mean={close_stats.get('mean', np.nan):.2f}, std={close_stats.get('std', np.nan):.2f}")
                        self.logger.info(f"NaN counts: {dict(df.isna().sum())}")
                        return df[required_columns]
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        bars = 20000
        self.logger.warning(f"Fallback: Fetching {bars} bars for {symbol}")
        async with mt5_lock:
            try:
                rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, bars)
            except Exception as e:
                self.logger.error(f"Fallback mt5.copy_rates_from_pos exception: {e}")
                rates = None

            if rates is not None and len(rates) >= min_bars:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df.set_index('time', inplace=True)
                required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
                if not all(col in df.columns for col in required_columns):
                    self.logger.error(f"Missing required columns in fallback data for {symbol}: {df.columns}")
                    return None
                if not pd.api.types.is_numeric_dtype(df['close']):
                    self.logger.error(f"Non-numeric 'close' column in fallback data for {symbol}")
                    return None
                if df['close'].nunique() < 2:
                    self.logger.error(f"Close prices are constant or nearly constant in fallback data: unique values={df['close'].nunique()}")
                    return None
                try:
                    df = df[df['tick_volume'] < df['tick_volume'].quantile(0.99)]
                    df = df[df['close'].pct_change().abs() < 0.1]
                    df = df[df['close'].between(df['close'].quantile(0.005), df['close'].quantile(0.995))]
                except Exception:
                    pass
                if len(df) >= min_bars:
                    close_stats = df['close'].describe()
                    self.logger.info(f"Fallback filtered data stats for {symbol}: rows={len(df)}, min={close_stats.get('min', np.nan):.2f}, max={close_stats.get('max', np.nan):.2f}, mean={close_stats.get('mean', np.nan):.2f}, std={close_stats.get('std', np.nan):.2f}")
                    self.logger.info(f"Fallback NaN counts: {dict(df.isna().sum())}")
                    data_cache[symbol] = df
                    return df[required_columns]
        self.logger.error(f"Failed to fetch sufficient data for {symbol} after {self.max_retries} attempts")
        return None

# ---------------------------------------------------------------------------
# FeatureEngineer (Item 9: max_window=50)
# ---------------------------------------------------------------------------
class FeatureEngineer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.max_indicator_window = 50  # Updated for SMA50 (Item 9)
        # Cache selected columns and fitted scaler from first full training call.
        # All subsequent calls reuse these so the feature set is stable.
        self._selected_columns    = None  # list[str] | None  -- post-T4 selected set
        self._all_feature_columns = None  # list[str] | None  -- full pre-T4 candidate set
        self._fitted_scaler       = None  # fitted scaler | None
        self._fitted_close_scaler = None

    def calculate_atr(self, df, period=14):
        if df is None or len(df) < period:
            self.logger.warning(f"Insufficient data for ATR: {len(df) if df is not None else 0} bars, need {period}")
            return pd.Series(np.nan, index=df.index if df is not None else [])
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=1).mean()
            return atr
        except Exception as e:
            self.logger.error(f"Error in ATR calculation: {e}")
            return pd.Series(np.nan, index=df.index)

    def prepare_features(self, df, lookback, symbol_type="forex"):
        min_required_bars = lookback + self.max_indicator_window
        if df is None or len(df) < min_required_bars:
            self.logger.error(f"Insufficient data for features: {len(df) if df is not None else 0} bars, need {min_required_bars}")
            return None, None, None, lookback, None

        df = df.copy()
        self.logger.info(f"Preparing features for {symbol_type}, initial rows: {len(df)}")
        try:
            if 'close' not in df.columns or not pd.api.types.is_numeric_dtype(df['close']):
                self.logger.error(f"Invalid or missing 'close' column in data: {df.columns}")
                return None, None, None, lookback, None
            if df['close'].var() < 1e-6:  # Low variance check
                self.logger.error("Close prices have insufficient variance")
                return None, None, None, lookback, None
            close_stats = df['close'].describe()
            self.logger.info(f"Close price stats before filtering: min={close_stats.get('min', np.nan):.2f}, max={close_stats.get('max', np.nan):.2f}, mean={close_stats.get('mean', np.nan):.2f}, std={close_stats.get('std', np.nan):.2f}")
            try:
                q_low = df['close'].quantile(0.005)
                q_high = df['close'].quantile(0.995)
                df_q = df[df['close'].between(q_low, q_high)]
                if len(df_q) >= min_required_bars:
                    df = df_q
                    self.logger.info(f"After close price quantile filter: {len(df)} rows")
                else:
                    self.logger.warning("Quantile filter would remove too many rows -> skipping.")
            except Exception as e:
                self.logger.warning(f"Quantile filter exception: {e}")

            if len(df) < min_required_bars:
                self.logger.error(f"Insufficient data after filtering: {len(df)} rows, need {min_required_bars}")
                return None, None, None, lookback, None
            if df['close'].nunique() < 2:
                self.logger.error(f"Close prices are constant or nearly constant: unique values={df['close'].nunique()}")
                return None, None, None, lookback, None

            try:
                df['rsi'] = RSIIndicator(df['close'], window=14).rsi().ffill().bfill()
            except Exception as e:
                self.logger.error(f"Error computing RSI: {e}")
                df['rsi'] = pd.Series(np.nan, index=df.index).ffill().bfill()

            try:
                macd_indicator = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['macd'] = macd_indicator.macd().ffill().bfill()
                df['macd_signal'] = macd_indicator.macd_signal().ffill().bfill()
            except Exception as e:
                self.logger.error(f"Error computing MACD: {e}")
                df['macd'] = pd.Series(np.nan, index=df.index).ffill().bfill()
                df['macd_signal'] = pd.Series(np.nan, index=df.index).ffill().bfill()

            try:
                df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().ffill().bfill()
            except Exception as e:
                self.logger.error(f"Error computing ADX: {e}")
                df['adx'] = pd.Series(np.nan, index=df.index).ffill().bfill()

            try:
                bb = BollingerBands(df['close'], window=20, window_dev=2)
                df['bb_upper'] = bb.bollinger_hband().ffill().bfill()
                df['bb_lower'] = bb.bollinger_lband().ffill().bfill()
            except Exception as e:
                self.logger.error(f"Error computing Bollinger Bands: {e}")
                df['bb_upper'] = pd.Series(np.nan, index=df.index).ffill().bfill()
                df['bb_lower'] = pd.Series(np.nan, index=df.index).ffill().bfill()

            try:
                # [T3] Session-reset VWAP: reset accumulator at the start of each trading day.
                # Falls back to a rolling 20-bar VWAP if the index has no date component.
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                typical_volume = df['tick_volume'] + 1e-10
                try:
                    session_date = df.index.normalize()
                    df['vwap'] = (
                        (typical_price * typical_volume)
                        .groupby(session_date).cumsum()
                        / typical_volume.groupby(session_date).cumsum()
                    )
                except Exception:
                    # Fallback: rolling 20-bar VWAP
                    df['vwap'] = (
                        (typical_price * typical_volume).rolling(20, min_periods=1).sum()
                        / typical_volume.rolling(20, min_periods=1).sum()
                    )
                df['vwap'] = df['vwap'].ffill().bfill()
            except Exception as e:
                self.logger.error(f"Error computing VWAP: {e}")
                df['vwap'] = pd.Series(np.nan, index=df.index).ffill().bfill()

            try:
                # [T2] Cyclic session/time-of-day features (sin/cos encoding preserves periodicity)
                hour = df.index.hour.astype(float)
                dow  = df.index.dayofweek.astype(float)
                df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
                df['dow_sin']  = np.sin(2 * np.pi * dow  / 5)
                df['dow_cos']  = np.cos(2 * np.pi * dow  / 5)
            except Exception as e:
                self.logger.error(f"Error computing time features: {e}")
                for col in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']:
                    df[col] = 0.0

            try:
                # Improvement 4: Price-action binary features
                # Engulfing: current body fully covers prior body
                body      = df['close'] - df['open']
                body_prev = body.shift(1)
                hi, lo    = df['high'], df['low']
                op, cl    = df['open'], df['close']
                df['pa_bull_engulf'] = (
                    (body > 0) & (body_prev < 0) &
                    (cl > op.shift(1)) & (op < cl.shift(1))
                ).astype(int)
                df['pa_bear_engulf'] = (
                    (body < 0) & (body_prev > 0) &
                    (cl < op.shift(1)) & (op > cl.shift(1))
                ).astype(int)
                # Pin bar: wick >= 2x body, closes near opposite end
                candle_range = (hi - lo).replace(0, np.nan)
                upper_wick   = hi - df[['open', 'close']].max(axis=1)
                lower_wick   = df[['open', 'close']].min(axis=1) - lo
                df['pa_bull_pin'] = (
                    (lower_wick >= 2 * abs(body)) &
                    (lower_wick / candle_range >= 0.6)
                ).astype(int)
                df['pa_bear_pin'] = (
                    (upper_wick >= 2 * abs(body)) &
                    (upper_wick / candle_range >= 0.6)
                ).astype(int)
                # Inside bar: high < prev high AND low > prev low
                df['pa_inside_bar'] = (
                    (hi < hi.shift(1)) & (lo > lo.shift(1))
                ).astype(int)
                for col in ['pa_bull_engulf', 'pa_bear_engulf', 'pa_bull_pin',
                            'pa_bear_pin', 'pa_inside_bar']:
                    df[col] = df[col].fillna(0).astype(int)
            except Exception as e:
                self.logger.warning(f"Price-action features failed: {e}")
                for col in ['pa_bull_engulf', 'pa_bear_engulf', 'pa_bull_pin',
                            'pa_bear_pin', 'pa_inside_bar']:
                    df[col] = 0

            try:
                stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
                df['stoch'] = stoch.stoch().ffill().bfill()
                df['stoch_signal'] = stoch.stoch_signal().ffill().bfill()
            except Exception as e:
                self.logger.error(f"Error computing Stochastic Oscillator: {e}")
                df['stoch'] = pd.Series(np.nan, index=df.index).ffill().bfill()
                df['stoch_signal'] = pd.Series(np.nan, index=df.index).ffill().bfill()

            df['return_1'] = df['close'].pct_change(1).ffill().bfill()
            df['return_2'] = df['close'].pct_change(2).ffill().bfill()
            df['volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std() * np.sqrt(252 * 16)
            df['volatility'] = df['volatility'].ffill().bfill()

            df['atr'] = self.calculate_atr(df, period=14).ffill().bfill()

            # Improved features
            df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator().ffill().bfill()
            df['sma50'] = SMAIndicator(df['close'], window=50).sma_indicator().ffill().bfill()
            df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci().ffill().bfill()
        except Exception as e:
            self.logger.error(f"Unexpected error in feature preparation: {e}")
            return None, None, None, lookback, None

        feature_columns = [
            'close', 'tick_volume', 'rsi', 'return_1', 'return_2', 'volatility',
            'macd', 'macd_signal', 'adx', 'bb_upper', 'bb_lower', 'vwap',
            'stoch', 'stoch_signal', 'atr', 'ema20', 'sma50', 'cci',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',  # [T2] cyclic time features
            'pa_bull_engulf', 'pa_bear_engulf',             # Improvement 4: price action
            'pa_bull_pin', 'pa_bear_pin', 'pa_inside_bar',
        ]
        if not all(col in df.columns for col in feature_columns):
            self.logger.error(f"Missing feature columns: {set(feature_columns) - set(df.columns)}")
            return None, None, None, lookback, None

        features = df[feature_columns].dropna()
        if features.empty:
            self.logger.error(f"No valid data after feature calculation for {feature_columns}")
            return None, None, None, lookback, None
        if 'close' not in features.columns or not pd.api.types.is_numeric_dtype(features['close']):
            self.logger.error(f"Invalid or missing 'close' column in features: {features.columns}")
            return None, None, None, lookback, None

        # [P1] Build direction labels: future_return over `predict_horizon` bars.
        # Classes: 0=down, 1=flat, 2=up.
        # predict_horizon should match your typical hold time (max_bars_open).
        # Defaults by symbol_type so gold/crypto (wide ATR) get different thresholds.
        _ph_defaults = {'forex': 6, 'stock': 5, 'crypto': 5}
        predict_horizon = self.config.get(
            'predict_horizon',
            _ph_defaults.get(symbol_type, 6)
        )
        # flat_thresh: fraction of ATR that separates "flat" from "up/down".
        # Narrower for forex (small ATR relative to price), wider for crypto.
        _ft_defaults = {'forex': 0.2, 'stock': 0.3, 'crypto': 0.5}
        flat_thresh_mult = self.config.get(
            'flat_thresh_mult',
            _ft_defaults.get(symbol_type, 0.3)
        )
        future_ret  = features['close'].pct_change(predict_horizon).shift(-predict_horizon)
        atr_rel     = features['atr'] / features['close']
        flat_thresh = flat_thresh_mult * atr_rel
        direction   = pd.Series(1, index=features.index, dtype=int)  # default flat
        direction[future_ret >  flat_thresh] = 2  # up
        direction[future_ret < -flat_thresh] = 0  # down
        # Drop the last predict_horizon rows where future_ret is NaN
        valid_mask = future_ret.notna()
        features   = features[valid_mask]
        direction  = direction[valid_mask]

        # Fix 7: log class distribution so imbalance is always visible in logs
        _counts = direction.value_counts().sort_index()
        _total  = len(direction)
        _dist   = {
            'down': f"{_counts.get(0, 0)} ({_counts.get(0, 0)/_total*100:.1f}%)",
            'flat': f"{_counts.get(1, 0)} ({_counts.get(1, 0)/_total*100:.1f}%)",
            'up':   f"{_counts.get(2, 0)} ({_counts.get(2, 0)/_total*100:.1f}%)",
        }
        self.logger.info(
            f"[Labels] {symbol_type} horizon={predict_horizon} flat_mult={flat_thresh_mult}: "
            f"down={_dist['down']}  flat={_dist['flat']}  up={_dist['up']}"
        )
        if _counts.get(1, 0) / _total > 0.65:
            self.logger.warning(
                f"[Labels] >65% flat labels -- model may predict flat constantly. "
                f"Consider reducing flat_thresh_mult in config.json."
            )

        scaler = RobustScaler()
        close_scaler = None  # Will be set in the scaler-cache block below
        close_prices = features['close'].copy()
        if close_prices.empty or close_prices.isna().all():
            self.logger.error(f"Empty or all-NaN close_prices Series: length={len(close_prices)}")
            return None, None, None, lookback, None
        # Temporary full-feature scaler used only for T4 importance ranking
        scaled_data = None
        try:
            scaled_data = scaler.fit_transform(features)
        except Exception as e:
            self.logger.warning("NaN in RobustScaler output, switching to MinMaxScaler")
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(features)
            if np.any(np.isnan(scaled_data)):
                self.logger.error(f"NaN detected in MinMaxScaler output for features: {feature_columns}")
                return None, None, None, lookback, None

        # [T4] Feature selection: prune bottom 30% by XGBoost feature importance.
        # MANDATORY columns are always kept regardless of importance score --
        # 'close' is used for current_price and VWAP; 'ema20'/'sma50' for trend
        # filter [E1]; 'atr' for SL sizing and exit thresholds.
        MANDATORY_COLS = {'close', 'ema20', 'sma50', 'atr'}

        # Expose the full pre-T4 candidate list so load_models can correctly check
        # whether cached columns are still computable from live data.  T4 may drop
        # 'macd' on one run and include it on another depending on importance scores;
        # the cache validity check must compare against ALL 22 computed columns, not
        # just the post-T4 subset that happened to be selected this run.
        self._all_feature_columns = list(feature_columns)
        if self._selected_columns is not None:
            # --- INFERENCE PATH: reuse columns selected during training ---
            missing = set(self._selected_columns) - set(features.columns)
            if missing:
                self.logger.error(
                    f"[T4] Inference column mismatch -- cached columns not in current data: {missing}. "
                    f"Cache is stale. Delete models/<symbol>/ and restart to retrain."
                )
                return None, None, None, lookback, None
            features = features[self._selected_columns]
            selected_columns = self._selected_columns
            self.logger.info(f"[T4] Reusing {len(selected_columns)} cached columns for inference")
        else:
            # --- TRAINING PATH: run selection once, cache result ---
            try:
                selector_xgb = xgb.XGBClassifier(
                    n_estimators=50, max_depth=4, learning_rate=0.1, n_jobs=1,
                    use_label_encoder=False, eval_metric='mlogloss', verbosity=0,
                    random_state=42,  # Fixed seed -> deterministic importance ranking
                )
                selector_xgb.fit(scaled_data, direction.values)
                importances = selector_xgb.feature_importances_
                threshold = np.percentile(importances, 30)
                selected_columns = [
                    c for c, imp in zip(feature_columns, importances)
                    if imp >= threshold or c in MANDATORY_COLS
                ]
                if len(selected_columns) < 5:
                    selected_columns = list(feature_columns)
                # Sort: mandatory cols first (stable), then rest alphabetically
                # This ensures the same column order across restarts even if importance
                # scores produce a different inclusion set due to data changes.
                mandatory_first = [c for c in selected_columns if c in MANDATORY_COLS]
                rest_sorted     = sorted(c for c in selected_columns if c not in MANDATORY_COLS)
                selected_columns = mandatory_first + rest_sorted
                features = features[selected_columns]
                self._selected_columns = selected_columns
                self.logger.info(f"[T4] Feature selection: {len(selected_columns)}/{len(feature_columns)} kept: {selected_columns}")
            except Exception as e:
                self.logger.warning(f"[T4] Feature selection failed, keeping all features: {e}")
                selected_columns = list(feature_columns)

        # Fit scaler on the final selected feature set.
        # Training path: fit fresh and cache. Inference path: return cached scaler.
        if self._fitted_scaler is not None:
            scaler = self._fitted_scaler
            close_scaler = self._fitted_close_scaler
        else:
            try:
                scaler = RobustScaler()
                scaler.fit(features)
            except Exception:
                scaler = MinMaxScaler()
                scaler.fit(features)
            # Re-fit close_scaler on selected features' close column
            close_scaler = MinMaxScaler()
            close_scaler.fit(features[['close']].values)
            self._fitted_scaler       = scaler
            self._fitted_close_scaler = close_scaler

        self.logger.info(f"Prepared data: features={list(features.columns)}, shape={features.shape}")
        try:
            self.logger.debug(f"Latest features sample: {dict(features.iloc[-1])}")
        except Exception:
            pass
        return features, scaler, close_scaler, lookback, direction

# ---------------------------------------------------------------------------
# ModelTrainer -- [P1] direction classification (3-class), [T1] walk-forward CV
# ---------------------------------------------------------------------------
class ModelTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def build_lstm_model(self, hp, lookback, n_features, n_classes=3):
        """[P1] 3-class direction classifier (up / flat / down)."""
        if hp is None:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(lookback, n_features)),
                tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.1),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64, recurrent_dropout=0.1),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(n_classes, activation='softmax'),
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
            return model

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(lookback, n_features)),
            tf.keras.layers.LSTM(
                units=hp.Int('units_1', 64, 256, step=32),
                return_sequences=True,
                recurrent_dropout=hp.Float('recurrent_dropout_1', 0.0, 0.3, step=0.1),
            ),
            tf.keras.layers.Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)),
            tf.keras.layers.LSTM(
                units=hp.Int('units_2', 32, 128, step=32),
                recurrent_dropout=hp.Float('recurrent_dropout_2', 0.0, 0.3, step=0.1),
            ),
            tf.keras.layers.Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)),
            tf.keras.layers.Dense(n_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    def train_lstm(self, X_train, y_train, X_val, y_val, lookback, n_features, symbol,
                   use_tuner=False, class_weight=None):
        self.logger.info(f"Starting LSTM direction training for {symbol} (use_tuner={use_tuner})")
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        if use_tuner:
            try:
                import keras_tuner as kt  # lazy import -- only needed when use_tuner=True
                tuner = kt.Hyperband(
                    lambda hp: self.build_lstm_model(hp, lookback, n_features),
                    objective='val_accuracy',
                    max_epochs=30,
                    directory=f'lstm_{symbol.lower()}',
                    project_name='tuner',
                )
                tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
                             callbacks=callbacks, class_weight=class_weight)
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = self.build_lstm_model(best_hps, lookback, n_features)
                model.fit(X_train, y_train, epochs=30, batch_size=32,
                          validation_data=(X_val, y_val), verbose=0,
                          callbacks=callbacks, class_weight=class_weight)
                self.logger.info(f"LSTM training (tuner) completed for {symbol}")
            except Exception as e:
                self.logger.error(f"Tuner failed: {e}, falling back to default model")
                model = self.build_lstm_model(None, lookback, n_features)
                model.fit(X_train, y_train, epochs=20, batch_size=32,
                          validation_data=(X_val, y_val), verbose=0,
                          callbacks=callbacks, class_weight=class_weight)
        else:
            model = self.build_lstm_model(None, lookback, n_features)
            model.fit(X_train, y_train, epochs=20, batch_size=32,
                      validation_data=(X_val, y_val), verbose=0,
                      callbacks=callbacks, class_weight=class_weight)
            self.logger.info(f"LSTM training (default) completed for {symbol}")
        return model

    def train_xgb(self, X_train, y_train, X_val, y_val, feature_columns, class_weight=None):
        """[P1] XGBoost direction classifier (multi:softmax, 3 classes).
        Fix 1: class_weight applied via sample_weight.
        Fix 3: TimeSeriesSplit replaces random 3-fold CV -- no temporal leakage.
        """
        self.logger.info("Starting XGBoost direction training")

        # Fix 1: compute per-sample weights from class_weight dict
        sample_weight_train = None
        if class_weight:
            sample_weight_train = np.array([class_weight.get(int(c), 1.0) for c in y_train])
            self.logger.info(f"[ClassWeight] XGBoost sample weights: {class_weight}")

        param_dist = {
            'n_estimators':     [200, 300, 400],
            'max_depth':        [4, 6, 8],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
        }
        cv_model = xgb.XGBClassifier(
            objective='multi:softmax', num_class=3, n_jobs=1,
            eval_metric='mlogloss', use_label_encoder=False, verbosity=0,
        )
        try:
            from sklearn.model_selection import RandomizedSearchCV as _RSCV, TimeSeriesSplit
            # Fix 3: TimeSeriesSplit respects temporal order -- no future data in validation
            tscv = TimeSeriesSplit(n_splits=5)
            random_search = _RSCV(cv_model, param_dist, n_iter=10, cv=tscv,
                                   scoring='accuracy', n_jobs=1)
            fit_kwargs = {}
            if sample_weight_train is not None:
                fit_kwargs['sample_weight'] = sample_weight_train
            random_search.fit(X_train, y_train, **fit_kwargs)
            best_params = random_search.best_params_
            self.logger.info(f"Best XGBoost params (TimeSeriesSplit): {best_params}")
            model = xgb.XGBClassifier(
                **best_params, objective='multi:softmax', num_class=3, n_jobs=1,
                early_stopping_rounds=10, use_label_encoder=False, verbosity=0,
            )
        except Exception as e:
            self.logger.warning(f"XGBoost RandomizedSearch failed: {e}, using defaults")
            model = xgb.XGBClassifier(
                objective='multi:softmax', num_class=3,
                n_estimators=200, max_depth=6, learning_rate=0.05,
                n_jobs=1, early_stopping_rounds=10,
                use_label_encoder=False, verbosity=0,
            )

        fit_kwargs = {'eval_set': [(X_val, y_val)], 'verbose': False}
        if sample_weight_train is not None:
            fit_kwargs['sample_weight'] = sample_weight_train
        model.fit(X_train, y_train, **fit_kwargs)

        try:
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            self.logger.info(f"XGBoost val accuracy: {acc:.4f}")
            # Fix 7: Log confusion matrix to catch "always predicts flat" pathology
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
            self.logger.info(
                f"Confusion matrix [down/flat/up]:\n"
                f"  Predicted->  down  flat  up\n"
                f"  True down:   {cm[0][0]:4d}  {cm[0][1]:4d}  {cm[0][2]:4d}\n"
                f"  True flat:   {cm[1][0]:4d}  {cm[1][1]:4d}  {cm[1][2]:4d}\n"
                f"  True up:     {cm[2][0]:4d}  {cm[2][1]:4d}  {cm[2][2]:4d}"
            )
            report = classification_report(y_val, y_pred, target_names=['down','flat','up'],
                                           zero_division=0)
            self.logger.info(f"Classification report:\n{report}")
            # Warn if the model is degenerate (predicts only one class)
            unique_pred = np.unique(y_pred)
            if len(unique_pred) == 1:
                self.logger.warning(
                    f"[ClassBalance] Model predicts only class {unique_pred[0]} -- "
                    f"likely class imbalance. Check flat_thresh_mult in config.json."
                )
            self.logger.info(f"Feature importances: {dict(zip(feature_columns, model.feature_importances_))}")
        except Exception as e:
            self.logger.warning(f"Failed to compute XGBoost val metrics: {e}")
        return model

    def walk_forward_train(self, X, y, lookback, n_features, symbol,
                           n_splits=5, use_tuner=False):
        """[T1] Walk-forward cross-validation.
        Fix 1: Computes balanced class weights from the full label distribution
        and passes them to both LSTM and XGBoost to counter class imbalance.
        """
        total = len(X)

        # Fix 1: compute balanced class weights from full dataset
        from sklearn.utils.class_weight import compute_class_weight
        try:
            classes      = np.array([0, 1, 2])
            cw_values    = compute_class_weight('balanced', classes=classes, y=y)
            class_weight = {int(c): float(w) for c, w in zip(classes, cw_values)}
            self.logger.info(f"[ClassWeight] Balanced weights: {class_weight}")
        except Exception as e:
            self.logger.warning(f"[ClassWeight] Could not compute weights: {e} -- using uniform")
            class_weight = None

        fold_size = total // (n_splits + 1)
        if fold_size < 50:
            self.logger.warning(f"[T1] Not enough data ({total} samples); falling back to single split")
            train_n = int(total * 0.8)
            return (
                self.train_lstm(X[:train_n], y[:train_n], X[train_n:], y[train_n:],
                                lookback, n_features, symbol, use_tuner,
                                class_weight=class_weight),
                self.train_xgb(X[:train_n, -1, :], y[:train_n],
                                X[train_n:, -1, :], y[train_n:],
                                [f'f{i}' for i in range(X.shape[2])],
                                class_weight=class_weight),
            )
        lstm_model, xgb_model = None, None
        for fold in range(n_splits):
            train_end = fold_size * (fold + 1)
            val_end   = min(train_end + fold_size, total)
            X_tr, y_tr = X[:train_end], y[:train_end]
            X_vl, y_vl = X[train_end:val_end], y[train_end:val_end]
            if len(X_vl) == 0:
                break
            self.logger.info(
                f"[T1] Walk-forward fold {fold+1}/{n_splits}: "
                f"train={len(X_tr)}, val={len(X_vl)}"
            )
            lstm_model = self.train_lstm(
                X_tr, y_tr, X_vl, y_vl, lookback, n_features, symbol,
                use_tuner=(use_tuner and fold == n_splits - 1),
                class_weight=class_weight,
            )
            xgb_model = self.train_xgb(
                X_tr[:, -1, :], y_tr, X_vl[:, -1, :], y_vl,
                [f'f{i}' for i in range(X.shape[2])],
                class_weight=class_weight,
            )
        return lstm_model, xgb_model

# ---------------------------------------------------------------------------
# Predictor -- [P1] direction classification, [P2] no hard clamp,
#             [P3] learned ensemble weights, [C1] ATR-normalised confidence,
#             [C2] no confidence floor
# ---------------------------------------------------------------------------
class Predictor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.lookback = config.get('lookback', 30)
        self.warmup_period = config.get('warmup_period', 20)
        self.predict_lstm = None
        # Recent predictions: (true_class, lstm_probs[3], xgb_class, ensemble_class)
        self.recent_predictions = deque(maxlen=self.warmup_period)
        # [P3] Learned ensemble weights (updated from recent accuracy)
        self._lstm_weight = 0.5
        self._xgb_weight  = 0.5

    def initialize_predictor(self, lstm_model):
        # Do NOT wrap in @tf.function: the decorator traces once per unique input
        # shape and locks the graph to that shape.  If T4 column count changes
        # between training runs the frozen trace produces a cryptic MatMul shape
        # mismatch at inference time.  Direct model.__call__ is fast enough for
        # single-batch inference and re-traces automatically on shape change.
        self._lstm_model   = lstm_model
        self.predict_lstm  = lambda inputs: lstm_model(inputs, training=False)
        self.logger.info("Initialised LSTM direction prediction function (no tf.function)")

    def _update_ensemble_weights(self):
        """[P3] Adjust LSTM vs XGBoost weight based on recent per-model accuracy."""
        if len(self.recent_predictions) < 10:
            return
        true_cls   = [p[0] for p in self.recent_predictions]
        lstm_cls   = [int(np.argmax(p[1])) for p in self.recent_predictions]
        xgb_cls    = [p[2] for p in self.recent_predictions]
        lstm_acc   = accuracy_score(true_cls, lstm_cls)
        xgb_acc    = accuracy_score(true_cls, xgb_cls)
        total      = lstm_acc + xgb_acc
        if total > 0:
            self._lstm_weight = lstm_acc / total
            self._xgb_weight  = xgb_acc / total
        self.logger.info(f"[P3] Ensemble weights updated: LSTM={self._lstm_weight:.2f}, XGB={self._xgb_weight:.2f}")

    def predict(self, lstm_model, xgb_model, data, scaler, close_scaler,
                feature_columns, current_price, symbol_type="forex", atr=None, symbol="?"):
        """Returns (action, confidence) where action in {'buy','sell','hold'}
        and confidence in [0, 1] (ATR-normalised accuracy proxy)."""
        min_required_bars = self.lookback + 50
        if len(data) < min_required_bars:
            self.logger.error(f"Insufficient data for prediction: {len(data)} bars, need {min_required_bars}")
            return 'hold', 0.0

        if self.predict_lstm is None:
            self.initialize_predictor(lstm_model)

        # Keep only known feature columns that exist in data, aligned to scaler's fitted columns
        try:
            # Get the exact column order the scaler was fitted on
            if hasattr(scaler, 'feature_names_in_'):
                fitted_cols = list(scaler.feature_names_in_)
            else:
                fitted_cols = [c for c in feature_columns if c in data.columns]

            # Check for genuinely missing columns -- these indicate a pipeline bug, not
            # just a small gap; filling with 0 would silently corrupt predictions.
            missing = set(fitted_cols) - set(data.columns)
            if missing:
                self.logger.error(
                    f"[Predict] Feature mismatch -- columns missing from data: {missing}. "
                    f"This usually means the model cache is stale. Delete models/{symbol}/ and restart."
                )
                return 'hold', 0.0

            # Align to exact column order, forward-fill genuine NaN gaps (weekend/holiday bars)
            aligned = data[fitted_cols].copy()
            # Only ffill/bfill for small gaps -- if >50% of a column is NaN it's a pipeline error
            for col in fitted_cols:
                nan_frac = aligned[col].isna().mean()
                if nan_frac > 0.5:
                    self.logger.error(
                        f"[Predict] Column '{col}' is {nan_frac:.0%} NaN -- pipeline error, skipping bar"
                    )
                    return 'hold', 0.0
                if aligned[col].isna().any():
                    aligned[col] = aligned[col].ffill().bfill()
            features = aligned
        except Exception as e:
            self.logger.error(f"Column alignment failed: {e}")
            return 'hold', 0.0

        try:
            scaled_data = scaler.transform(features)
        except Exception as e:
            self.logger.error(f"Scaler transform failed: {e}")
            return 'hold', 0.0

        if np.any(np.isnan(scaled_data)):
            self.logger.error("NaN in scaled data -- skipping prediction")
            return 'hold', 0.0

        X_lstm = np.array([scaled_data[-self.lookback:]])
        X_xgb  = np.array([scaled_data[-1]])

        # LSTM: softmax output -> class probabilities
        try:
            lstm_probs = self.predict_lstm(X_lstm)[0].numpy()  # shape (3,)
        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            lstm_probs = np.array([1/3, 1/3, 1/3])

        # XGBoost: hard class prediction
        try:
            xgb_class = int(xgb_model.predict(X_xgb)[0])
        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {e}")
            xgb_class = 1  # flat

        # [P3] Weighted ensemble: blend LSTM probs with XGBoost one-hot
        self._update_ensemble_weights()
        xgb_one_hot = np.zeros(3)
        xgb_one_hot[xgb_class] = 1.0
        ensemble_probs = self._lstm_weight * lstm_probs + self._xgb_weight * xgb_one_hot
        ensemble_class = int(np.argmax(ensemble_probs))   # 0=down, 1=flat, 2=up
        confidence_raw = float(ensemble_probs[ensemble_class])  # max prob

        # [C1] ATR-normalised confidence proxy
        # Use per-model agreement as a multiplier: both agree -> full confidence
        lstm_class = int(np.argmax(lstm_probs))
        agreement_bonus = 0.15 if lstm_class == xgb_class else 0.0
        # [C2] No floor -- confidence can be 0 and block the trade
        confidence = min(1.0, confidence_raw + agreement_bonus)

        class_to_action = {0: 'sell', 1: 'hold', 2: 'buy'}
        action = class_to_action[ensemble_class]

        self.logger.info(
            f"Direction prediction: LSTM={lstm_probs.round(3)}, XGB={xgb_class}, "
            f"ensemble={ensemble_class}({action}), conf={confidence:.3f}"
        )
        # Cache for run() to feed into record_outcome / recent_predictions
        self._last_lstm_probs = lstm_probs
        self._last_xgb_class  = xgb_class
        self._last_ens_class  = ensemble_class
        return action, confidence

    def record_outcome(self, true_class, lstm_probs, xgb_class, ensemble_class):
        """Call after the bar closes to feed the accuracy tracker."""
        self.recent_predictions.append((true_class, lstm_probs, xgb_class, ensemble_class))

    def confidence_ok(self, confidence, min_confidence=0.45):
        """[C1][C2] True when confidence clears the threshold (no floor bypass)."""
        return confidence >= min_confidence

# ---------------------------------------------------------------------------
# TradingAgent (Item 10: Slippage/comm in rewards, Item 15: Adaptive binning)
# ---------------------------------------------------------------------------
class TradingAgent:
    def __init__(self, config, logger, symbol, normalized_symbol, symbol_type):
        self.config = config
        self.logger = logger
        self.symbol = symbol
        self.normalized_symbol = normalized_symbol
        self.symbol_type = symbol_type
        self.q_table = self.load_q_table()
        self.actions = ['buy', 'sell', 'hold']
        self.epsilon = config.get('epsilon', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.9)
        self.position = None  # {'type': 'buy' or 'sell', 'entry': price, 'volume': volume, 'partial_closed': False}
        # [E3] Reduced state space: 3 features x 5 bins (was 6 x 10)
        self._state_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

    def load_q_table(self):
        if platform.system() == "Emscripten":
            self.logger.warning(f"Q-table loading not supported in Pyodide for {self.symbol}")
            return {}
        filename = f"q_table_{self.normalized_symbol}.pkl"
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    q_table = pickle.load(f)
                self.logger.info(f"Q-table loaded for {self.symbol}")
                return q_table
            except Exception as e:
                self.logger.error(f"Failed to load Q-table {filename}: {e}")
                return {}
        return {}

    def save_q_table(self):
        if platform.system() == "Emscripten":
            self.logger.warning(f"Q-table saving not supported in Pyodide for {self.symbol}")
            return
        filename = f"q_table_{self.normalized_symbol}.pkl"
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            self.logger.info(f"Q-table saved for {self.symbol}")
        except Exception as e:
            self.logger.error(f"Failed to save Q-table {filename}: {e}")

    def discretize_state(self, features):
        """[E3] 3-feature x 5-bin state space.
        Gracefully handles T4 pruning: uses whichever of rsi/adx/macd are present."""
        # Preferred columns in priority order; use whatever survived T4
        candidates = [c for c in ['rsi', 'adx', 'macd', 'volatility', 'atr'] if c in features.columns]
        if not candidates:
            self.logger.warning("discretize_state: no usable columns -- returning zero state")
            return (0, 0, 0)
        # Use up to 3 columns for the state tuple
        state_cols = candidates[:3]
        # Pad with the first column if fewer than 3 survived
        while len(state_cols) < 3:
            state_cols.append(state_cols[0])

        try:
            scalar_vals = [float(features[c].iloc[-1]) for c in state_cols]
        except Exception as e:
            self.logger.warning(f"discretize_state scalar read failed: {e}")
            return (0, 0, 0)

        bins = 5
        state_data = features[state_cols].iloc[-50:]
        try:
            # Re-fit discretizer only on the available columns
            disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
            discretized = disc.fit_transform(state_data).astype(int)[-1]
            return tuple(int(v) for v in discretized)
        except Exception as e:
            self.logger.warning(f"Adaptive binning failed: {e} -- using fixed bins")
            result = []
            ranges = {'rsi': (0, 100), 'adx': (0, 100)}
            for col, val in zip(state_cols, scalar_vals):
                lo, hi = ranges.get(col, (state_data[col].min(), state_data[col].max()))
                result.append(int(np.digitize(val, np.linspace(lo, hi, bins))))
            return tuple(result)

    def choose_action(self, state, predicted_action, features):
        """Improvement 6: rule-based entry filter replaces the sparse Q-table.

        The Q-table was unlikely to converge -- with only a handful of trades per
        session it could never fill 125 states.  The rule-based filter is simpler,
        fully debuggable, and produces the same or better selectivity.

        Config key  use_q_table: true  restores the old Q-table behaviour.

        Rule-based logic:
          - Compute an entry score from confidence (passed externally) +
            ADX strength + trend alignment margin (EMA20-SMA50 spread).
          - Require score >= entry_score_threshold (default 0.0 -- just the
            trend/ADX gates already applied in E1/E2).
          - Epsilon exploration still fires but is still gated by E1/E2 below.
        """
        use_q_table = self.config.get('use_q_table', False)
        # Confidence override: skip E1/E2 filters when model is very certain
        confidence_override = self.config.get('confidence_override_threshold', 0.90)

        if use_q_table:
            q_values = [self.q_table.get((state, a), 0.0) for a in self.actions]
            all_zero  = all(v == 0.0 for v in q_values)
            if np.random.random() < self.epsilon:
                # Fix issue 4: epsilon exploration must not go against the model's
                # direction. Only explore between predicted_action and 'hold',
                # never pick the opposite direction randomly.
                safe_explore = [predicted_action, 'hold'] if predicted_action != 'hold' \
                               else ['hold']
                action = np.random.choice(safe_explore)
                self.logger.debug(f"[Q/epsilon] Safe explore: {action} (model={predicted_action})")
            else:
                q_action = self.actions[int(np.argmax(q_values))]
                action   = predicted_action if all_zero else q_action
        else:
            if np.random.random() < self.epsilon:
                safe_explore = [predicted_action, 'hold'] if predicted_action != 'hold' \
                               else ['hold']
                action = np.random.choice(safe_explore)
                self.logger.debug(f"[Rule/epsilon] Safe explore: {action} (model={predicted_action})")
            else:
                action = predicted_action

        # [E1] Trend filter -- skip if disabled OR confidence override applies
        trend_filter = self.config.get('trend_filter_enabled', True)
        _confidence  = getattr(self, '_last_confidence', 0.0)
        _override    = _confidence >= confidence_override and confidence_override > 0
        if _override and action != 'hold':
            self.logger.info(
                f"[ConfOverride] {self.symbol}: bypassing E1/E2 filters "
                f"(confidence={_confidence:.3f} >= {confidence_override})"
            )
        if trend_filter and not _override:
            try:
                ema20 = features['ema20'].iloc[-1]
                sma50 = features['sma50'].iloc[-1]
                if action == 'buy'  and ema20 <= sma50:
                    if self.config.get('log_filter_blocks', True):
                        self.logger.info(
                            f"[E1/TrendFilter] {self.symbol}: BUY blocked "
                            f"(EMA20={ema20:.4f} <= SMA50={sma50:.4f})"
                        )
                    action = 'hold'
                elif action == 'sell' and ema20 >= sma50:
                    if self.config.get('log_filter_blocks', True):
                        self.logger.info(
                            f"[E1/TrendFilter] {self.symbol}: SELL blocked "
                            f"(EMA20={ema20:.4f} >= SMA50={sma50:.4f})"
                        )
                    action = 'hold'
            except Exception as e:
                self.logger.warning(f"[E1] Trend filter failed: {e}")

        # [E2] ADX filter -- skip if disabled OR confidence override applies
        adx_filter = self.config.get('adx_filter_enabled', True)
        if adx_filter and not _override and action != 'hold':
            try:
                adx_col = next((c for c in ['adx'] if c in features.columns), None)
                if adx_col:
                    adx     = features[adx_col].iloc[-1]
                    adx_min = self.config.get('adx_min_entry', 18)
                    if adx < adx_min:
                        if self.config.get('log_filter_blocks', True):
                            self.logger.info(
                                f"[E2/ADXFilter] {self.symbol}: {action} blocked "
                                f"(ADX={adx:.1f} < {adx_min})"
                            )
                        action = 'hold'
            except Exception as e:
                self.logger.warning(f"[E2] ADX filter failed: {e}")

        self.logger.info(f"Action for {self.symbol}: {action} (mode={'Q-table' if use_q_table else 'rule-based'})")
        return action

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action), 0.0)
        next_max_q = max([self.q_table.get((next_state, a), 0.0) for a in self.actions])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    async def calculate_reward(self, action, entry_price, current_price, volume, predicted_action=None):
        if action == "hold":
            return 0.0
        async with mt5_lock:
            symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"Cannot retrieve symbol info for {self.symbol}")
            return 0.0
        point = symbol_info.point
        pip_value = symbol_info.trade_tick_value if symbol_info.trade_tick_value is not None else 1.0
        try:
            if action == "buy":
                profit = (current_price - entry_price) / point * pip_value * volume
            else:
                profit = (entry_price - current_price) / point * pip_value * volume
        except Exception as e:
            self.logger.warning(f"Profit calc failed: {e}")
            profit = 0.0

        # Slippage and commission estimate
        slippage_points = 2
        commission = symbol_info.commission if hasattr(symbol_info, 'commission') and symbol_info.commission else 0.5 * volume
        profit -= (slippage_points * point * pip_value * volume) + commission

        # Direction-accuracy bonus/penalty
        if predicted_action is not None:
            correct = (action == 'buy'  and predicted_action == 'buy') or \
                      (action == 'sell' and predicted_action == 'sell')
            direction_bonus = 0.1 * abs(profit) if correct else -0.1 * abs(profit)
            return profit + direction_bonus
        return profit

# ---------------------------------------------------------------------------
# TradeExecutor (Item 2: Harden anti-pyramiding, Item 4: math.floor volume, Item 5: Event-driven trailing, Item 20: Limit/stop orders)
# ---------------------------------------------------------------------------
class TradeExecutor:
    def __init__(self, config, logger, data_fetcher, feature_engineer):
        self.config = config
        self.logger = logger
        self.data_fetcher = data_fetcher
        self.feature_engineer = feature_engineer
        self.magic_number = None  # Set per symbol in run

    async def execute_trade(self, symbol, action, volume, predicted_action, current_price, recent_lstm_rmse, recent_xgb_rmse, trading_agent, symbol_type, atr=None):
        if action == "hold":
            return True, None, None, None
        async with mt5_lock:
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Cannot retrieve symbol info for {symbol}")
            return False, None, None, None
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level * point
        self.logger.info(f"Trade stops level for {symbol}: {stops_level:.2f} points")
        for _ in range(5):
            async with mt5_lock:
                tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                break
            await asyncio.sleep(1)
        if tick is None:
            self.logger.error(f"Cannot retrieve tick data for {symbol}")
            return False, None, None, None
        price = tick.ask if action == "buy" else tick.bid

        # Use caller-supplied ATR (already from full feature window); only fall back if None
        if atr is None:
            atr_data = await self.data_fetcher.fetch_data(symbol, bars=500)
            atr_series = self.feature_engineer.calculate_atr(atr_data)
            if atr_series is None or atr_series.empty:
                self.logger.error(f"ATR calculation failed for {symbol}")
                return False, None, None, None
            atr = atr_series.iloc[-1]
        if pd.isna(atr) or atr <= 0:
            self.logger.error(f"ATR invalid ({atr}) for {symbol}")
            return False, None, None, None

        ensemble_rmse = (recent_lstm_rmse + recent_xgb_rmse) / 2  # kept for RR scaling only; gating moved to run()

        # Improvement 1: Volatility Tiers for TP
        volatility_factor = atr / current_price  # Normalize
        if volatility_factor < 0.001:
            rr_tier = 1.5  # Low vol: tighter TP
        elif volatility_factor > 0.003:
            rr_tier = 3.0  # High vol: wider TP
        else:
            rr_tier = 2.0  # Normal

        # [S1] SL multiplier is symbol-type aware; stocks are noisier than forex
        sl_atr_multiplier = self.config.get('sl_atr_multiplier', {}).get(
            symbol_type, 1.5 if symbol_type == 'forex' else (2.5 if symbol_type == 'crypto' else 2.0)
        )
        rr_multiplier = rr_tier

        sl_points = atr * sl_atr_multiplier / point
        sl_points = max(sl_points, symbol_info.trade_stops_level)
        tp_points = sl_points * rr_multiplier
        if action == "buy":
            sl = price - sl_points * point
            tp = price + tp_points * point
        else:
            sl = price + sl_points * point
            tp = price - tp_points * point

        min_distance = max(atr * 0.5 / point, symbol_info.trade_stops_level)
        if action == "buy":
            if (tp - price) / point < min_distance:
                tp = price + min_distance * point
            if (price - sl) / point < min_distance:
                sl = price - min_distance * point
        elif action == "sell":
            if (price - tp) / point < min_distance:
                tp = price - min_distance * point
            if (sl - price) / point < min_distance:
                sl = price + min_distance * point

        # Fix: Snap volumes to valid steps (Item 4)
        volume_step = symbol_info.volume_step
        if volume_step <= 0:
            volume_step = 0.01 if symbol_type != "stock" else 1.0
        # volume_step is now guaranteed > 0; snap helper avoids repeating this logic
        def snap(v):
            snapped = math.floor(v / volume_step) * volume_step
            return max(symbol_info.volume_min, min(symbol_info.volume_max, snapped))

        # Improvement 1: Partial TP Setup
        partial_enabled = self.config.get('partial_tp', {}).get('enabled', True)
        partial_percent = self.config.get('partial_tp', {}).get('percent', 0.5)
        partial_rr = self.config.get('partial_tp', {}).get('rr', 1.0)

        if partial_enabled:
            partial_volume = snap(volume * partial_percent)
            full_volume = snap(volume - partial_volume)
            # If full_volume rounds below minimum, consolidate into partial
            if full_volume < symbol_info.volume_min:
                partial_volume = snap(volume)
                full_volume = 0
        else:
            partial_volume = snap(volume)
            full_volume = 0

        self.logger.info(f"Snapped volumes for {symbol}: total={volume}, partial={partial_volume}, full={full_volume}, step={volume_step}")

        # Send main position with full TP
        order_type = mt5.ORDER_TYPE_BUY if action == "buy" else mt5.ORDER_TYPE_SELL
        # Always use market orders; limit orders require confidence_factor which lives in run()
        trade_action = mt5.TRADE_ACTION_DEAL
        def _try_filling_modes(request_base):
            """Try ORDER_FILLING_IOC -> FOK -> RETURN until one succeeds.
            Returns (result, used_filling) or (None, None) if all fail."""
            modes = [
                ("IOC",    mt5.ORDER_FILLING_IOC),
                ("FOK",    mt5.ORDER_FILLING_FOK),
                ("RETURN", mt5.ORDER_FILLING_RETURN),
            ]
            for name, mode in modes:
                req = {**request_base, "type_filling": mode}
                res = mt5.order_send(req)
                if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                    self.logger.info(f"Order filled with mode {name} for {symbol}")
                    return res, mode
                self.logger.warning(f"Filling mode {name} failed for {symbol}: {res.comment if res else 'None'} (retcode={res.retcode if res else 'N/A'})")
            return None, None

        request_base = {
            "action":    trade_action,
            "symbol":    symbol,
            "volume":    partial_volume,
            "type":      order_type,
            "price":     price,
            "sl":        sl,
            "tp":        tp,
            "magic":     self.magic_number,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        async with mt5_lock:
            result_full, used_filling = _try_filling_modes(request_base)

        if result_full is None:
            self.logger.error(f"Failed to open main position for {symbol}: all filling modes rejected")
            return False, None, None, None

        deal = result_full.deal
        self.logger.info(f"Main position opened: deal={deal}, volume={partial_volume}, TP={tp:.5f}")

        success = True
        if partial_enabled and full_volume > 0:
            partial_tp_points = sl_points * partial_rr
            partial_tp = price + partial_tp_points * point if action == "buy" else price - partial_tp_points * point
            partial_base = {**request_base, "volume": full_volume, "tp": partial_tp}
            async with mt5_lock:
                result_partial, _ = _try_filling_modes(partial_base)
            if result_partial is None:
                self.logger.warning(f"Partial TP order failed for {symbol} after all filling modes -- closing main")
                await self.close_position(symbol, trading_agent)
                return False, None, None, None
            self.logger.info(f"Partial TP position opened: volume={full_volume}, TP={partial_tp:.5f}")

        if success:
            self.logger.info(f"Trade executed for {symbol}: {action}, volume={volume}, price={price:.6f}, SL={sl:.6f}, TP={tp:.6f} (RR={rr_multiplier:.2f})")
            trading_agent.position = {'type': action, 'entry': price, 'volume': volume, 'partial_closed': False}
        return success, deal, sl, action if success else None

    async def get_actual_pnl(self, symbol, position_tickets, since_timestamp):
        """Read actual closed P&L from MT5 deal history for all position tickets.
        Accepts a list of tickets (main order + partial TP order).
        Returns sum of all deal profits, or None if history unavailable.
        """
        if not position_tickets:
            return None
        try:
            await asyncio.sleep(1)  # let MT5 settle deals into history
            dt_from = datetime.fromtimestamp(
                min(since_timestamp) - 5 if isinstance(since_timestamp, list) else since_timestamp - 5,
                tz=timezone.utc
            ).replace(tzinfo=None)
            dt_to = (datetime.now(timezone.utc) + timedelta(seconds=10)).replace(tzinfo=None)
            async with mt5_lock:
                deals = mt5.history_deals_get(dt_from, dt_to)
            if deals is None:
                return None
            ticket_set = set(position_tickets) if isinstance(position_tickets, list) \
                         else {position_tickets}
            pnl = sum(
                d.profit for d in deals
                if d.position_id in ticket_set
                and d.entry == mt5.DEAL_ENTRY_OUT
            )
            self.logger.info(
                f"[ActualPnL] {symbol} tickets={ticket_set}: total P&L={pnl:.4f} "
                f"from {len([d for d in deals if d.position_id in ticket_set])} deals"
            )
            return pnl if (pnl != 0 or any(
                d.position_id in ticket_set and d.entry == mt5.DEAL_ENTRY_OUT
                for d in deals
            )) else None
        except Exception as e:
            self.logger.warning(f"get_actual_pnl failed for {symbol}: {e}")
            return None

    async def close_position(self, symbol, trading_agent):
        async with mt5_lock:
            positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            self.logger.debug(f"No open positions for {symbol}")
            return False, [], []
        success         = False
        closed_tickets  = []
        open_timestamps = []
        for position in positions:
            if getattr(position, 'magic', None) != self.magic_number:
                continue
            type_opposite = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            base_req = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "symbol":    symbol,
                "volume":    position.volume,
                "type":      type_opposite,
                "position":  position.ticket,
                "magic":     self.magic_number,
                "type_time": mt5.ORDER_TIME_GTC,
            }
            async with mt5_lock:
                for name, mode in [("IOC", mt5.ORDER_FILLING_IOC), ("FOK", mt5.ORDER_FILLING_FOK), ("RETURN", mt5.ORDER_FILLING_RETURN)]:
                    req = {**base_req, "type_filling": mode}
                    result = mt5.order_send(req)
                    if result is not None and getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(
                            f"Closed position for {symbol}: ticket={position.ticket}, "
                            f"vol={position.volume}, filling={name}"
                        )
                        success = True
                        closed_tickets.append(position.ticket)
                        open_timestamps.append(
                            getattr(position, 'time', int(datetime.now(timezone.utc).timestamp()))
                        )
                        break
                    self.logger.warning(
                        f"Close filling mode {name} failed: {result.comment if result else 'None'}"
                    )
        if success:
            trading_agent.position = None
        return success, closed_tickets, open_timestamps

    async def update_trailing_stop(self, symbol):
        async with mt5_lock:
            positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            self.logger.debug(f"No open positions for {symbol}")
            return False
        async with mt5_lock:
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Cannot retrieve symbol info for {symbol}")
            return False
        point = symbol_info.point

        # ATR from cache
        atr = None
        if symbol in data_cache and len(data_cache[symbol]) >= 20:
            try:
                atr_series = self.feature_engineer.calculate_atr(data_cache[symbol])
                if atr_series is not None and not atr_series.empty:
                    atr = float(atr_series.iloc[-1])
            except Exception as e:
                self.logger.warning(f"ATR from cache failed for {symbol}: {e}")
        if atr is None or pd.isna(atr) or atr <= 0:
            self.logger.warning(f"ATR unavailable for trailing stop {symbol} -- using fallback")
            atr = symbol_info.point * 100

        # Fix 1: config-driven trailing multiplier (separate from initial SL multiplier
        # so you can trail tighter without widening the entry SL)
        symbol_type = (
            "crypto" if any(s in symbol.upper() for s in ["BTC","ETH","XRP","LTC","SOL"])
            else "stock" if any(s in symbol.upper() for s in ["TSLA","NVDA",".CASH","#"])
            else "forex"
        )
        trail_mult = self.config.get('trailing_atr_multiplier', {}).get(
            symbol_type,
            1.0 if symbol_type == 'forex' else (1.5 if symbol_type == 'crypto' else 1.2)
        )
        base_sl_points = atr * trail_mult / point

        # Fix 2: no new-bar gate -- trailing fires every poll cycle (every 3-10 min)
        # The old 900s gate disabled trailing for most of every H4 candle.

        # Fix 3: fresh tick with correct bid/ask per direction
        for _ in range(5):
            async with mt5_lock:
                tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                break
            await asyncio.sleep(1)
        if tick is None:
            self.logger.error(f"Cannot retrieve tick data for {symbol}")
            return False

        success = False
        for position in positions:
            if getattr(position, 'magic', None) != self.magic_number:
                continue
            current_sl  = getattr(position, 'sl', 0)
            entry_price = position.price_open
            is_buy      = position.type == mt5.POSITION_TYPE_BUY

            # Fix 3: correct price per direction
            current_price = tick.bid if is_buy else tick.ask

            profit_points = (
                (current_price - entry_price) / point if is_buy
                else (entry_price - current_price) / point
            )
            atr_points = atr / point

            # Breakeven: move SL to entry once profit >= breakeven_atr_trigger x ATR
            be_trigger = self.config.get('breakeven_atr_trigger', 1.0)
            if profit_points >= atr_points * be_trigger:
                be_sl       = entry_price
                sl_improved = (is_buy and be_sl > current_sl) or (not is_buy and (current_sl == 0 or be_sl < current_sl))
                if sl_improved:
                    async with mt5_lock:
                        result = mt5.order_send({
                            "action":   mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "symbol":   symbol,
                            "sl":       be_sl,
                            "tp":       position.tp,
                        })
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(
                            f"[Breakeven] {symbol} SL -> entry {be_sl:.5f} "
                            f"(profit={profit_points:.1f} pts >= {be_trigger}x ATR)"
                        )
                        success = True

            # Fix 4: step-based tightening -- trail closer as profit grows
            #   profit < 1x ATR  -> base_sl_points (full trail distance)
            #   profit >= 1x ATR -> 0.75x base     (tighter)
            #   profit >= 2x ATR -> 0.5x base      (locks in most profit)
            if profit_points >= atr_points * 2.0:
                sl_points = base_sl_points * 0.5
            elif profit_points >= atr_points * 1.0:
                sl_points = base_sl_points * 0.75
            else:
                sl_points = base_sl_points

            # Enforce minimum distance (broker requirement)
            min_dist  = max(atr * 0.3 / point, symbol_info.trade_stops_level)
            sl_points = max(sl_points, min_dist)

            if is_buy:
                new_sl = current_price - sl_points * point
                if new_sl > current_sl and new_sl < current_price:
                    async with mt5_lock:
                        result = mt5.order_send({
                            "action":   mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "symbol":   symbol,
                            "sl":       new_sl,
                            "tp":       position.tp,
                        })
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(
                            f"[Trail] {symbol} BUY SL {current_sl:.5f} -> {new_sl:.5f} "
                            f"(profit={profit_points:.1f} pts, dist={sl_points:.1f} pts)"
                        )
                        success = True
                    else:
                        self.logger.warning(
                            f"[Trail] {symbol} BUY failed: "
                            f"{getattr(result,'comment','') if result else 'no response'}"
                        )
            else:
                new_sl = current_price + sl_points * point
                if (new_sl < current_sl or current_sl == 0) and new_sl > current_price:
                    async with mt5_lock:
                        result = mt5.order_send({
                            "action":   mt5.TRADE_ACTION_SLTP,
                            "position": position.ticket,
                            "symbol":   symbol,
                            "sl":       new_sl,
                            "tp":       position.tp,
                        })
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(
                            f"[Trail] {symbol} SELL SL {current_sl:.5f} -> {new_sl:.5f} "
                            f"(profit={profit_points:.1f} pts, dist={sl_points:.1f} pts)"
                        )
                        success = True
                    else:
                        self.logger.warning(
                            f"[Trail] {symbol} SELL failed: "
                            f"{getattr(result,'comment','') if result else 'no response'}"
                        )
        return success

# ---------------------------------------------------------------------------
# RiskManager (Item 3: Dynamic Kelly, Item 6: Per-symbol equity, Item 7: Accurate daily-loss)
# ---------------------------------------------------------------------------
class RiskManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.risk_percent      = config.get('risk', {}).get('risk_percent', 1.5)
        self.daily_loss_limit  = config.get('risk', {}).get('daily_loss_limit', 5.0)
        self.max_drawdown      = config.get('risk', {}).get('max_drawdown', 15.0)
        # Account-level TP: stop trading for the day once this profit (in account currency)
        # is reached. 0 or None = disabled.
        self.daily_profit_target = config.get('risk', {}).get('daily_profit_target', 0)
        self.magic_number = None
        self.peak_equity  = 0
        self.win_prob     = 0.55
        self.rr_ratio     = 1.5
        self.history      = deque(maxlen=100)

    async def calculate_lot_size(self, symbol, account_equity, symbol_type, current_price, atr):
        async with mt5_lock:
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Cannot retrieve symbol info for {symbol}")
            return 0.01
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level * point
        self.logger.info(f"Trade stops level for {symbol}: {stops_level:.2f} points")
        if pd.isna(atr):
            self.logger.error(f"ATR calculation failed for {symbol}")
            return float(symbol_info.volume_min)
        # [S1] Read sl_atr_multiplier from config (same as execute_trade) so lot size
        # is consistent with the actual SL distance that will be placed.
        sl_atr_multiplier = self.config.get('sl_atr_multiplier', {}).get(
            symbol_type, 1.5 if symbol_type == 'forex' else (2.5 if symbol_type == 'crypto' else 2.0)
        )
        sl_pips = atr * sl_atr_multiplier / point
        sl_pips = max(sl_pips, symbol_info.trade_stops_level)
        # Per-symbol equity (Item 6): divide total equity by number of concurrently traded symbols
        num_symbols = self.config.get('num_symbols', 1)
        allocated_equity = account_equity / max(1, num_symbols)
        risk_amount = allocated_equity * (self.risk_percent / 100)
        pip_value = symbol_info.trade_tick_value if symbol_info.trade_tick_value is not None else 1.0
        if sl_pips == 0 or pip_value == 0:
            self.logger.warning("SL pips or pip value is zero, returning minimum lot")
            return float(symbol_info.volume_min)
        # vol_adj scales the risk budget for volatility environment
        vol_adj = max(0.5, min(2.0, 0.001 / (atr / current_price)))
        # Kelly fraction applied to the vol-adjusted budget (not compounded on lot_size)
        if len(self.history) > 20:
            wins = sum(1 for p in self.history if p > 0)
            self.win_prob = wins / len(self.history)
            avg_win  = sum(abs(p) for p in self.history if p > 0) / max(1, wins)
            avg_loss = sum(abs(p) for p in self.history if p < 0) / max(1, len(self.history) - wins)
            self.rr_ratio = avg_win / max(1e-6, avg_loss)
        kelly_f = self.win_prob - (1 - self.win_prob) / self.rr_ratio
        kelly_f = max(0.01, min(0.2, kelly_f))
        # Apply both adjustments to risk_amount before sizing -- single multiplication path
        risk_amount = risk_amount * vol_adj * kelly_f
        lot_size = risk_amount / (sl_pips * pip_value)
        self.logger.info(f"Kelly={kelly_f:.2f}, vol_adj={vol_adj:.2f}, risk_amount={risk_amount:.2f}")
        # Fix: Snap to volume_step
        volume_step = symbol_info.volume_step
        if volume_step <= 0:
            volume_step = 0.01 if symbol_type != "stock" else 1.0
        if volume_step > 0:
            num_steps = math.floor(lot_size / volume_step)
            lot_size = num_steps * volume_step
        lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
        self.logger.info(f"Calculated lot size for {symbol}: {lot_size} (min: {symbol_info.volume_min}, max: {symbol_info.volume_max}, step: {symbol_info.volume_step}, atr={atr}, sl_pips={sl_pips}, vol_adj={vol_adj:.2f})")
        return lot_size

    async def check_limits(self, account_equity, account_balance=None):
        """Check all account-level risk and profit limits.

        Returns one of:
          'ok'          -- all clear, continue trading
          'stop'        -- daily loss / drawdown exceeded; stop new trades, keep positions
          'close_all'   -- equity profit threshold hit; close all positions immediately
        """
        self.peak_equity = max(self.peak_equity, account_equity)
        drawdown = max(0, (self.peak_equity - account_equity) / self.peak_equity * 100)
        if drawdown > self.max_drawdown:
            self.logger.error(f"Max drawdown limit ({self.max_drawdown}%) exceeded: {drawdown:.2f}%")
            return 'stop'

        # --- Equity-level floating profit threshold ---
        # If (equity - balance) >= equity_profit_threshold, lock in profit by
        # closing all open positions immediately.
        equity_profit_threshold = self.config.get('risk', {}).get('equity_profit_threshold', 0)
        if equity_profit_threshold and account_balance is not None:
            floating_profit = account_equity - account_balance
            if floating_profit >= equity_profit_threshold:
                self.logger.info(
                    f"[AccountTP] Equity profit threshold reached: "
                    f"equity={account_equity:.2f}, balance={account_balance:.2f}, "
                    f"floating={floating_profit:.2f} >= {equity_profit_threshold:.2f} "
                    f"-- closing all positions now"
                )
                return 'close_all'

        # --- Broker-time daily history ---
        async with mt5_lock:
            terminal_info = mt5.terminal_info()
        if terminal_info is not None and getattr(terminal_info, 'server_time', 0):
            broker_now = datetime.fromtimestamp(terminal_info.server_time, tz=timezone.utc)
        else:
            broker_now = datetime.now(timezone.utc)
            self.logger.debug("check_limits: terminal_info unavailable, falling back to local UTC")

        broker_today_midnight = broker_now.replace(hour=0, minute=0, second=0, microsecond=0)
        dt_from = broker_today_midnight.replace(tzinfo=None)
        dt_to   = (broker_today_midnight + timedelta(days=1)).replace(tzinfo=None)

        async with mt5_lock:
            history = mt5.history_deals_get(dt_from, dt_to)

        if history is None:
            error_code = mt5.last_error()
            self.logger.warning(
                f"history_deals_get failed (broker_now={broker_now.isoformat()}, "
                f"dt_from={dt_from}, dt_to={dt_to}, mt5_error={error_code}) -- "
                f"skipping daily-loss check for this bar"
            )
            return 'ok'

        total_profit = sum(
            deal.profit for deal in history
            if getattr(deal, 'magic', None) == self.magic_number
            and deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]
            and deal.entry == mt5.DEAL_ENTRY_OUT
        )
        self.logger.debug(
            f"Daily P&L check: {len(history)} deals fetched, "
            f"magic-filtered profit={total_profit:.2f}"
        )

        # Daily loss limit (% of equity)
        loss_percent = max(0.0, -total_profit / max(account_equity, 1) * 100)
        if loss_percent > self.daily_loss_limit:
            self.logger.error(
                f"Daily loss limit ({self.daily_loss_limit}%) exceeded: {loss_percent:.2f}%"
            )
            return 'stop'

        # Daily closed profit target (account currency)
        if self.daily_profit_target and total_profit >= self.daily_profit_target:
            self.logger.info(
                f"[AccountTP] Daily profit target reached: "
                f"{total_profit:.2f} >= {self.daily_profit_target:.2f} -- "
                f"no new trades until next session"
            )
            return 'stop'

        return 'ok'

    def update_history(self, profit):
        self.history.append(profit)  # deque maxlen=100 handles eviction automatically

# ---------------------------------------------------------------------------
# MarketChecker (Item 8: gmt_offset)
# ---------------------------------------------------------------------------
class MarketChecker:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.time_zone_offset_hours = config.get('time_zone_offset_hours', -5)

    async def is_market_open(self, symbol):
        """Check whether the market is currently open for trading.

        Priority:
          1. MT5 copy_rates_from_pos -- if broker returns data, market is live
          2. MT5 symbol session schedule (from symbol_info_tick staleness check)
          3. Time-based fallback:
               - Crypto (BTC/ETH etc): 7/24 -- no weekend blackout
               - Forex/Gold:           closed Fri >=21:00 UTC to Sun <21:00 UTC
               - Stocks:               09:30-16:00 broker-local, Mon-Fri only
        """
        # ------------------------------------------------------------------
        # Step 1: MT5 trade_mode -- broker authoritative
        # ------------------------------------------------------------------
        async with mt5_lock:
            sym_info  = mt5.symbol_info(symbol)
            term_info = mt5.terminal_info()

        if sym_info is not None:
            if sym_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                self.logger.info(
                    f"Market closed for {symbol}: MT5 trade_mode={sym_info.trade_mode}"
                )
                return False

        # ------------------------------------------------------------------
        # Step 2: determine server time (UTC)
        # ------------------------------------------------------------------
        try:
            server_time = datetime.fromtimestamp(term_info.server_time, tz=timezone.utc)
            gmt_offset  = term_info.gmt_offset / 3600
        except Exception:
            async with mt5_lock:
                tick = mt5.symbol_info_tick(symbol)
            if tick is not None and getattr(tick, 'time', None):
                server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
                gmt_offset  = self.time_zone_offset_hours
                self.logger.warning(
                    f"Using tick time for {symbol} due to missing terminal_info.server_time"
                )
            else:
                server_time = datetime.now(timezone.utc)
                gmt_offset  = self.time_zone_offset_hours
                self.logger.warning(
                    f"Using local UTC time for {symbol} (no terminal/tick time available)"
                )

        utc_now  = server_time
        dow      = utc_now.weekday()      # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        utc_hour = utc_now.hour + utc_now.minute / 60.0

        self.logger.info(
            f"Server time: {utc_now.isoformat()}  "
            f"broker_local: {utc_now.astimezone(timezone(timedelta(hours=gmt_offset))).strftime('%a %H:%M')} UTC"
        )

        # ------------------------------------------------------------------
        # Step 3: classify symbol type
        # ------------------------------------------------------------------
        is_stock  = any(s in symbol.upper() for s in ["TSLA", "NVDA", ".CASH", "#"])
        is_crypto = any(s in symbol.upper() for s in ["BTC", "ETH", "XRP", "LTC", "SOL"])

        # ------------------------------------------------------------------
        # Step 4a: stocks -- session-bound, Mon-Fri only
        # ------------------------------------------------------------------
        if is_stock:
            broker_time  = utc_now.astimezone(timezone(timedelta(hours=gmt_offset)))
            market_open  = broker_time.replace(hour=9,  minute=30, second=0, microsecond=0)
            market_close = broker_time.replace(hour=16, minute=0,  second=0, microsecond=0)
            is_open = (market_open <= broker_time <= market_close) and dow < 5
            self.logger.info(
                f"Market {'open' if is_open else 'closed'} for {symbol}: "
                f"stock session (dow={dow}, broker={broker_time.strftime('%H:%M')})"
            )
            return is_open

        # ------------------------------------------------------------------
        # Step 4b: crypto -- respect broker session schedule from MT5
        # If broker says trade_mode=FULL (checked above) AND symbol has
        # sessions on Sat/Sun (like BTCUSD on your broker), treat as 7/24.
        # We detect this by checking if the symbol has a Saturday session.
        # ------------------------------------------------------------------
        if is_crypto:
            # Check if broker publishes a Saturday trading session for this symbol.
            # If yes, treat as 7/24 -- no weekend blackout applied.
            has_weekend_session = False
            try:
                async with mt5_lock:
                    # Saturday session check: dow=5 in MT5 session_info is index 6 (Sun=0...Sat=6)
                    sat_sessions = mt5.symbol_info_tick(symbol)
                    # Simpler: if sym_info.trade_mode is FULL and it's a crypto, trust it
                    if sym_info is not None and sym_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                        has_weekend_session = True
            except Exception:
                pass

            if has_weekend_session:
                self.logger.info(f"Market open for {symbol}: 7/24 crypto trading")
                return True

            # Crypto with unknown session -- fall through to forex rules as safe default

        # ------------------------------------------------------------------
        # Step 4c: forex / gold -- 24/5 with weekend blackout
        #
        # Closed:  Friday  >= forex_close_fri_utc  (default 21:00 UTC)
        #          Saturday -- all day
        #          Sunday   <  forex_open_sun_utc   (default 21:00 UTC)
        # ------------------------------------------------------------------
        close_fri_utc = self.config.get('forex_close_fri_utc', 21)
        open_sun_utc  = self.config.get('forex_open_sun_utc',  21)

        if dow == 5:
            self.logger.info(f"Market closed for {symbol}: Saturday")
            return False
        if dow == 6 and utc_hour < open_sun_utc:
            self.logger.info(
                f"Market closed for {symbol}: Sunday {utc_now.strftime('%H:%M')} UTC "
                f"(opens ~{open_sun_utc:02d}:00 UTC)"
            )
            return False
        if dow == 4 and utc_hour >= close_fri_utc:
            self.logger.info(
                f"Market closed for {symbol}: Friday {utc_now.strftime('%H:%M')} UTC "
                f"(closed from {close_fri_utc:02d}:00 UTC)"
            )
            return False

        self.logger.info(
            f"Market open for {symbol}: 24/5 trading "
            f"(UTC {utc_now.strftime('%a %H:%M')})"
        )
        return True

# ---------------------------------------------------------------------------
# Backtester (Item 17: Viz)
# ---------------------------------------------------------------------------
class Backtester:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.lookback = config.get('lookback', 30)
        self.initial_capital = config.get('initial_capital', 10000)

    async def run_backtest(self, symbol, symbol_type):
        data = await DataFetcher(self.config, self.logger).fetch_data(symbol, bars=10000)
        if data is None:
            self.logger.error(f"Cannot backtest {symbol}: No data")
            return None
        fe = FeatureEngineer(self.config, self.logger)
        lookback = min(len(data) - fe.max_indicator_window, self.lookback)
        # [P1][T2][T3][T4] new 5-return signature
        features, scaler, close_scaler, lookback, direction = fe.prepare_features(data, lookback, symbol_type)
        if features is None:
            self.logger.error(f"Cannot backtest {symbol}: Feature preparation failed")
            return None
        model_trainer = ModelTrainer(self.config, self.logger)
        X_seq, y_seq = [], []
        dir_vals = direction.values
        for i in range(lookback, len(features)):
            X_seq.append(scaler.transform(features.iloc[i - lookback:i]))
            y_seq.append(dir_vals[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        if len(X_seq) < 80:
            self.logger.error(f"Insufficient data for backtest training: {len(X_seq)} samples")
            return None
        # [T1] Walk-forward training for backtest too
        lstm_model, xgb_model = model_trainer.walk_forward_train(
            X_seq, y_seq, lookback, len(features.columns), symbol,
            n_splits=self.config.get('wf_splits', 5)
        )
        if lstm_model is None:
            self.logger.error(f"Backtest model training failed for {symbol}")
            return None

        predictor    = Predictor(self.config, self.logger)
        predictor.initialize_predictor(lstm_model)
        trading_agent = TradingAgent(self.config, self.logger, symbol, symbol.replace("+", ""), symbol_type)

        profits         = []
        cash            = self.initial_capital
        position        = None
        portfolio_values = []
        bars_in_pos     = 0
        max_bars_open   = self.config.get('max_bars_open', 8)  # [S3]

        test_start = int(len(X_seq) * 0.8)
        for i in range(test_start, len(X_seq)):
            current_price  = features['close'].iloc[i]
            window_features = features.iloc[max(0, i - lookback):i]
            atr_series      = fe.calculate_atr(features.iloc[max(0, i - 20):i + 1])
            atr             = atr_series.iloc[-1] if (atr_series is not None and not atr_series.empty) else None

            predicted_action, confidence = predictor.predict(
                lstm_model, xgb_model, window_features, scaler, close_scaler,
                list(features.columns), current_price, symbol_type, atr=atr, symbol=symbol
            )

            # Manage open position
            if position is not None:
                bars_in_pos += 1
                should_close = False
                # [S2] Direction reversal exit
                if position['type'] == 'buy'  and predicted_action == 'sell': should_close = True
                if position['type'] == 'sell' and predicted_action == 'buy':  should_close = True
                # [S3] Time stop
                if bars_in_pos >= max_bars_open: should_close = True

                if should_close:
                    if position['type'] == 'buy':
                        cash = position['volume'] * current_price
                    else:
                        cash = position['volume'] * (2 * position['entry'] - current_price)
                    position   = None
                    bars_in_pos = 0

            # New entry
            if position is None and confidence >= self.config.get('min_confidence', 0.45):
                state  = trading_agent.discretize_state(window_features)
                action = trading_agent.choose_action(state, predicted_action, window_features)
                if action != 'hold':
                    volume   = cash / current_price
                    position = {'type': action, 'entry': current_price, 'volume': volume}
                    bars_in_pos = 0
                    if action == 'buy':
                        cash -= volume * current_price
                    else:
                        cash += volume * current_price

            pv = cash + (position['volume'] * current_price if position and position['type'] == 'buy' else 0)
            portfolio_values.append(pv)
            profits.append(pv - (portfolio_values[-2] if len(portfolio_values) > 1 else self.initial_capital))

        if not portfolio_values:
            self.logger.error(f"No portfolio values computed for {symbol}")
            return None
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital * 100
        sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252) if np.std(profits) != 0 else 0
        win_count    = sum(1 for p in profits if p > 0)
        win_rate     = win_count / len(profits) * 100 if profits else 0
        self.logger.info(
            f"Backtest {symbol}: Return={total_return:.2f}%, "
            f"Sharpe={sharpe_ratio:.2f}, WinRate={win_rate:.1f}%"
        )
        trading_agent.save_q_table()
        # Viz -- lazy import so startup time is not penalised for live trading
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(portfolio_values, label='Portfolio Value')
        ax.set_title(f"Equity Curve for {symbol}")
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()
        fig.savefig(f"equity_curve_{symbol}.png")
        plt.close(fig)  # Prevent figure memory leak across multiple backtest calls
        self.logger.info(f"Equity curve saved as equity_curve_{symbol}.png")
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_values': portfolio_values,
            'profits': profits
        }

# ---------------------------------------------------------------------------
# TradeJournal -- improvement 10: per-symbol CSV trade log
# ---------------------------------------------------------------------------
class TradeJournal:
    """Appends one row per closed trade to trades_<symbol>.csv.
    Columns: timestamp, symbol, direction, entry, exit, sl, tp,
             volume, pnl, confidence, adx_at_entry, reason, bars_held
    """
    def __init__(self, symbol):
        safe = symbol.replace("+", "plus").replace(".", "_").replace("/", "_")
        self.path = f"trades_{safe}.csv"
        self._header = [
            "timestamp", "symbol", "direction", "entry", "exit",
            "sl", "tp", "volume", "pnl", "confidence",
            "adx_at_entry", "reason", "bars_held",
        ]
        if not os.path.exists(self.path):
            import csv
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._header)

    def log(self, *, symbol, direction, entry, exit_price, sl, tp,
            volume, pnl, confidence, adx_at_entry, reason, bars_held):
        import csv
        row = [
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            symbol, direction,
            round(entry, 6), round(exit_price, 6),
            round(sl, 6) if sl else "", round(tp, 6) if tp else "",
            round(volume, 4), round(pnl, 4),
            round(confidence, 4), round(adx_at_entry, 2) if adx_at_entry else "",
            reason, bars_held,
        ]
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception:
            pass  # Journal failure must never crash the bot


# ---------------------------------------------------------------------------
# TradingBot (Item 12: UUID & shutdown, Item 18: Alerts, Item 19: Sentiment skipped as high effort)
# ---------------------------------------------------------------------------
class TradingBot:
    def __init__(self, config_file="config.json"):
        self.logger = Logger()
        self.config = Config(config_file)
        self.data_fetcher = DataFetcher(self.config, self.logger)
        # Dedicated H4 fetcher for MTF confirmation -- never mutates the main fetcher's
        # timeframe, eliminating the race condition when multiple symbols run concurrently.
        _h4_config = type('_C', (), {
            'get': lambda self_, k, default=None, symbol=None: (
                'TIMEFRAME_H4' if k == 'timeframe' else self.config.get(k, default, symbol=symbol)
            )
        })()
        self._h4_fetcher = DataFetcher(_h4_config, self.logger)
        self.feature_engineer = FeatureEngineer(self.config, self.logger)
        self.model_trainer = ModelTrainer(self.config, self.logger)
        self.predictor = Predictor(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)
        self.market_checker = MarketChecker(self.config, self.logger)
        self.trade_executor = TradeExecutor(self.config, self.logger, self.data_fetcher, self.feature_engineer)
        self.lstm_model = None
        self.xgb_model = None
        self.last_train_bars = 0
        self.last_rmse = 0.0  # For model drift detection
        # NOTE: signal handlers are registered once in the main thread (see bottom of file)
        # Registering here would crash when TradingBot is created inside a worker thread.

    async def send_alert(self, message):
        """Send alert via Telegram (preferred) and/or email (fallback).
        Configure in config.json or environment variables:
          Telegram: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
          Email:    SMTP_HOST, SMTP_PORT, SMTP_FROM, SMTP_TO, SMTP_USER, SMTP_PASSWORD
        """
        sent = False
        # --- Telegram (recommended: fast, reliable, no SMTP setup needed) ---
        tg_token   = os.environ.get("TELEGRAM_TOKEN",   self.config.get("telegram_token", ""))
        tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID", self.config.get("telegram_chat_id", ""))
        if tg_token and tg_chat_id:
            try:
                import urllib.request, urllib.parse
                payload = urllib.parse.urlencode({
                    "chat_id": tg_chat_id,
                    "text":    f"[TradingBot] {message}",
                    "parse_mode": "HTML",
                }).encode()
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{tg_token}/sendMessage",
                    data=payload, method="POST"
                )
                urllib.request.urlopen(req, timeout=5)
                self.logger.info(f"Telegram alert sent: {message}")
                sent = True
            except Exception as e:
                self.logger.warning(f"Telegram alert failed: {e}")

        # --- Email (fallback) ---
        smtp_host = os.environ.get("SMTP_HOST", self.config.get("smtp_host", ""))
        smtp_port = int(os.environ.get("SMTP_PORT", self.config.get("smtp_port", 587)))
        smtp_from = os.environ.get("SMTP_FROM", self.config.get("smtp_from", ""))
        smtp_to   = os.environ.get("SMTP_TO",   self.config.get("smtp_to",   ""))
        smtp_user = os.environ.get("SMTP_USER", self.config.get("smtp_user", ""))
        smtp_pass = os.environ.get("SMTP_PASSWORD", self.config.get("smtp_password", ""))
        if smtp_host and smtp_to:
            try:
                msg = MIMEText(message)
                msg['Subject'] = 'Trading Bot Alert'
                msg['From']    = smtp_from
                msg['To']      = smtp_to
                with smtplib.SMTP(smtp_host, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                self.logger.info(f"Email alert sent: {message}")
                sent = True
            except Exception as e:
                self.logger.error(f"Email alert failed: {e}")
        if not sent:
            self.logger.debug(f"No alert channel configured; message: {message}")

    async def normalize_symbol(self, symbol):
        normalized = ''.join(c for c in symbol if c.isalnum()).lower()
        async with mt5_lock:
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None and getattr(symbol_info, 'trade_contract_size', 0) > 0:
            self.logger.info(f"Symbol {symbol} is tradable")
            return symbol, normalized, True
        self.logger.warning(f"Symbol {symbol} not found or not tradable, checking alternatives")
        async with mt5_lock:
            symbols = mt5.symbols_get()
        if symbols is None:
            self.logger.error(f"Failed to retrieve symbols list for {symbol}")
            return None, None, False
        similar_symbols = [s.name for s in symbols if symbol.upper() in s.name.upper() or symbol.replace("+", "").upper() in s.name.upper()]
        if not similar_symbols:
            self.logger.error(f"No similar symbols found for {symbol}")
            return None, None, False
        self.logger.info(f"Similar symbols for {symbol}: {similar_symbols}")
        fallback_symbol = "XAUUSD" if "XAUUSD" in symbol.upper() else None
        if fallback_symbol:
            async with mt5_lock:
                symbol_info = mt5.symbol_info(fallback_symbol)
            if symbol_info and getattr(symbol_info, 'trade_contract_size', 0) > 0:
                self.logger.info(f"Using fallback symbol {fallback_symbol} for {symbol}")
                return fallback_symbol, normalized, True
        for sim_symbol in similar_symbols:
            async with mt5_lock:
                symbol_info = mt5.symbol_info(sim_symbol)
            if symbol_info and getattr(symbol_info, 'trade_contract_size', 0) > 0:
                self.logger.info(f"Using similar symbol {sim_symbol} for {symbol}")
                return sim_symbol, normalized, True
        self.logger.error(f"No tradable symbol found for {symbol}. Check broker's Market Watch.")
        return None, None, False

    def _model_dir(self, symbol):
        """Return a per-symbol directory for persisted model artefacts."""
        safe = symbol.replace("+", "plus").replace(".", "_").replace("/", "_")
        d = os.path.join("models", safe)
        os.makedirs(d, exist_ok=True)
        return d

    def save_models(self, symbol, scaler, selected_columns):
        """Persist LSTM, XGBoost, scaler and feature columns to disk."""
        if not selected_columns:
            self.logger.warning(f"[ModelCache] No selected_columns to save for {symbol} -- skipping")
            return
        d = self._model_dir(symbol)
        try:
            self.lstm_model.save(os.path.join(d, "lstm.keras"))
            self.xgb_model.save_model(os.path.join(d, "xgb.json"))
            import pickle, hashlib as _hl
            # Fingerprint = MD5 of the sorted column list so we can detect T4 drift on reload
            col_fingerprint = _hl.md5(",".join(sorted(selected_columns)).encode()).hexdigest()
            with open(os.path.join(d, "scaler.pkl"), "wb") as f:
                pickle.dump({
                    "scaler":           scaler,
                    "selected_columns": selected_columns,
                    "col_fingerprint":  col_fingerprint,
                    "n_features":       len(selected_columns),
                    "close_scaler":     self.feature_engineer._fitted_close_scaler,
                    "saved_at":         datetime.now(timezone.utc).isoformat(),
                }, f)
            self.logger.info(f"[ModelCache] Saved to {d}/  cols={selected_columns}  fingerprint={col_fingerprint[:8]}")
        except Exception as e:
            self.logger.warning(f"[ModelCache] Save failed for {symbol}: {e}")

    def _probe_cached_columns(self, symbol):
        """Read only scaler.pkl to get the saved column list -- fast, no model weights.
        Returns list[str] if a valid, non-stale pickle exists, else None."""
        import pickle
        scaler_path = os.path.join(self._model_dir(symbol), "scaler.pkl")
        if not os.path.exists(scaler_path):
            return None
        try:
            with open(scaler_path, "rb") as f:
                meta = pickle.load(f)
            max_age_h = self.config.get("model_max_age_hours", 24)
            saved_at  = datetime.fromisoformat(meta["saved_at"])
            age_h     = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600
            if age_h > max_age_h:
                return None
            cols = meta.get("selected_columns")
            return cols if cols else None
        except Exception as e:
            self.logger.warning(f"[ModelCache] _probe_cached_columns failed for {symbol}: {e}")
            return None

    def load_models(self, symbol, current_columns=None):
        """Load persisted models if they exist, are within max_age_hours, AND have the
        same feature columns as the current dataset.  Returns
        (lstm, xgb, scaler, close_scaler, selected_columns) or all-None on any mismatch."""
        import pickle, hashlib as _hl
        none5 = (None,) * 5
        d = self._model_dir(symbol)
        lstm_path   = os.path.join(d, "lstm.keras")
        xgb_path    = os.path.join(d, "xgb.json")
        scaler_path = os.path.join(d, "scaler.pkl")

        if not all(os.path.exists(p) for p in [lstm_path, xgb_path, scaler_path]):
            self.logger.info(f"[ModelCache] No saved models for {symbol} -- training fresh")
            return none5

        try:
            with open(scaler_path, "rb") as f:
                meta = pickle.load(f)
        except Exception as e:
            self.logger.warning(f"[ModelCache] Cannot read scaler.pkl for {symbol}: {e} -- retraining")
            return none5

        # 1. Age check
        max_age_h = self.config.get("model_max_age_hours", 24)
        try:
            saved_at = datetime.fromisoformat(meta["saved_at"])
            age_h = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600
        except Exception:
            age_h = float('inf')
        if age_h > max_age_h:
            self.logger.info(f"[ModelCache] Models for {symbol} are {age_h:.1f}h old (limit {max_age_h}h) -- retraining")
            return none5

        # 2. Column fingerprint check -- if T4 would produce a different column set, retrain
        cached_cols = meta.get("selected_columns", [])
        if current_columns is not None:
            # current_columns = full candidate set produced by prepare_features BEFORE T4
            # We check whether the cached selected set is a valid subset of what's available now
            missing_in_current = set(cached_cols) - set(current_columns)
            if missing_in_current:
                self.logger.info(
                    f"[ModelCache] Cached columns no longer available in live data "
                    f"{missing_in_current} -- retraining {symbol}"
                )
                return none5
            # Also compare fingerprints if both are stored
            if "col_fingerprint" in meta:
                live_fp = _hl.md5(",".join(sorted(cached_cols)).encode()).hexdigest()
                if live_fp != meta["col_fingerprint"]:
                    self.logger.info(f"[ModelCache] Column fingerprint mismatch for {symbol} -- retraining")
                    return none5

        # 3. Load models
        try:
            lstm_model = tf.keras.models.load_model(lstm_path)
            xgb_model  = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)
        except Exception as e:
            self.logger.warning(f"[ModelCache] Model load failed for {symbol}: {e} -- retraining")
            return none5

        # 4. Shape consistency check: LSTM input width must match the number of
        #    cached columns.  A mismatch means the model and scaler were saved in
        #    different training runs (e.g. a partial save after a crash) and would
        #    produce a cryptic MatMul shape error at inference time.
        try:
            lstm_n_features = lstm_model.input_shape[-1]  # (batch, lookback, n_features)
            expected = meta.get("n_features", len(cached_cols))
            if lstm_n_features != expected or lstm_n_features != len(cached_cols):
                self.logger.warning(
                    f"[ModelCache] Shape mismatch for {symbol}: LSTM expects "
                    f"{lstm_n_features} features, cache has {len(cached_cols)} cols, "
                    f"saved n_features={expected} -- retraining"
                )
                return none5
        except Exception as e:
            self.logger.warning(f"[ModelCache] Could not verify LSTM shape for {symbol}: {e}")

        self.logger.info(
            f"[ModelCache] Loaded models for {symbol} -- age {age_h:.1f}h, "
            f"cols={cached_cols}"
        )
        return lstm_model, xgb_model, meta["scaler"], meta["close_scaler"], cached_cols

    async def train_models(self, symbol, features, scaler, lookback, direction=None,
                           use_tuner=None, selected_columns=None):
        """[P1] Train direction classifiers. [T1] Uses walk-forward CV.
        features must already be T4-selected; pass selected_columns explicitly."""
        if use_tuner is None:
            use_tuner = self.config.get('use_tuner', False)

        if direction is None:
            self.logger.error(f"No direction labels provided for {symbol}; cannot train")
            return False

        # If caller didn't pass selected_columns, fall back to what FeatureEngineer cached.
        # Do NOT reset the cache here -- features is already the post-T4 DataFrame.
        if selected_columns is None:
            selected_columns = self.feature_engineer._selected_columns
        if selected_columns is None:
            selected_columns = list(features.columns)

        X_seq, y_seq = [], []
        dir_vals = direction.values
        for i in range(lookback, len(features)):
            X_seq.append(scaler.transform(features.iloc[i - lookback:i]))
            y_seq.append(dir_vals[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        if len(X_seq) < 80:
            self.logger.warning(f"Low data for training: {len(X_seq)} samples, need at least 80")
            if len(X_seq) < 50:
                self.logger.error(f"Insufficient data for training: {len(X_seq)} samples")
                return False

        # [T1] Walk-forward CV; also returns final-fold models
        lstm_m, xgb_m = self.model_trainer.walk_forward_train(
            X_seq, y_seq, lookback, len(features.columns), symbol,
            n_splits=self.config.get('wf_splits', 5), use_tuner=use_tuner
        )
        if lstm_m is None or xgb_m is None:
            self.logger.error(f"walk_forward_train returned None for {symbol}")
            return False

        self.lstm_model = lstm_m
        self.xgb_model  = xgb_m
        self.predictor.initialize_predictor(self.lstm_model)
        self.last_train_bars = len(features)
        self.logger.info(f"Models trained for {symbol} with {len(X_seq)} direction samples (use_tuner={use_tuner})")

        # Persist to disk -- use the resolved selected_columns, not _selected_columns which
        # may still be None if the cache reset happened before T4 ran in prepare_features.
        self.save_models(symbol, scaler, selected_columns)
        return True

    async def run(self, symbol, backtest=False):
        if not self.data_fetcher.initialize_mt5():
            self.logger.error("Failed to initialize MT5, exiting")
            return
        symbol, normalized_symbol, is_tradable = await self.normalize_symbol(symbol)
        if symbol is None or not is_tradable:
            self.logger.error(f"Cannot trade {symbol}: Symbol not tradable or not found")
            return
        symbol_type = (
            "stock"  if any(s in symbol.upper() for s in ["TSLA", "NVDA", ".CASH", "#"]) else
            "crypto" if any(s in symbol.upper() for s in ["BTC", "ETH", "XRP", "LTC", "SOL"]) else
            "forex"
        )
        self.logger.info(f"Symbol type for {symbol}: {symbol_type}")

        # Magic number: use config override if present, otherwise derive deterministically
        # from the symbol name via MD5. Same symbol -> same number across restarts.
        # Override in config.json:  "XAUUSD": { "magic_number": 123456 }
        configured_magic = self.config.get('magic_number', None, symbol=symbol)
        if configured_magic:
            self.magic_number = int(configured_magic)
        else:
            _hash_bytes = hashlib.md5(symbol.encode()).digest()
            self.magic_number = 100000 + (int.from_bytes(_hash_bytes[:4], 'big') % 900000)
        self.trade_executor.magic_number = self.magic_number
        self.risk_manager.magic_number   = self.magic_number
        self.risk_manager.peak_equity    = self.config.get('initial_capital', 10000)
        self.logger.info(f"Magic number for {symbol}: {self.magic_number} ({'config' if configured_magic else 'auto-derived'})")

        if backtest:
            self.logger.info(f"Running backtest for {symbol}")
            backtester = Backtester(self.config, self.logger)
            await backtester.run_backtest(symbol, symbol_type)
            return

        trading_agent    = TradingAgent(self.config, self.logger, symbol, normalized_symbol, symbol_type)
        lookback         = self.config.get('lookback', 30, symbol=symbol)
        retrain_interval = self.config.get('retrain_interval', 100)
        max_bars_open    = self.config.get('max_bars_open', 8)
        min_confidence   = self.config.get('min_confidence', 0.45)
        last_trade_time  = 0
        trade_open_bar   = None
        # Improvement 10: trade journal
        journal = TradeJournal(symbol)
        # Track per-trade metadata for the journal
        _entry_confidence = 0.0
        _entry_adx        = 0.0
        _entry_sl         = 0.0
        _entry_tp         = 0.0

        # --- initial data fetch ---
        data = await self.data_fetcher.fetch_data(symbol, bars=lookback + 5000)
        if data is None:
            self.logger.error(f"Failed to fetch initial data for {symbol}, exiting")
            return

        # --- PROBE cache before running prepare_features so T4 never re-selects
        #     columns when a valid cache already exists. ---
        # Step 1: quick probe -- load scaler.pkl only (fast, no model weights)
        _cols_probe = self._probe_cached_columns(symbol)
        if _cols_probe is not None:
            # Inject the cached column list so prepare_features takes the inference
            # path (skips T4) and produces features aligned to the saved model.
            self.feature_engineer._selected_columns = _cols_probe
            self.logger.info(f"[ModelCache] Pre-injecting {len(_cols_probe)} cached columns before prepare_features")

        # Step 2: prepare features (T4 skipped if _selected_columns was injected above)
        features, scaler, close_scaler, adjusted_lookback, direction = \
            self.feature_engineer.prepare_features(data, lookback, symbol_type)
        if features is None:
            # If inference path failed (column not available), clear cache and retry fresh
            self.logger.warning(f"[ModelCache] Cached column set incompatible with live data -- clearing and retraining")
            self.feature_engineer._selected_columns   = None
            self.feature_engineer._fitted_scaler       = None
            self.feature_engineer._fitted_close_scaler = None
            features, scaler, close_scaler, adjusted_lookback, direction = \
                self.feature_engineer.prepare_features(data, lookback, symbol_type)
            if features is None:
                self.logger.error(f"Failed to prepare initial features for {symbol}, exiting")
                return
            _cols_probe = None  # Force full retrain below

        # Step 3: load full model weights (only if column probe succeeded)
        if _cols_probe is not None:
            pre_t4_columns = self.feature_engineer._all_feature_columns or list(features.columns)
            lstm_cached, xgb_cached, scaler_cached, close_scaler_cached, cols_cached = \
                self.load_models(symbol, current_columns=pre_t4_columns)
        else:
            lstm_cached = None

        if lstm_cached is not None:
            self.lstm_model  = lstm_cached
            self.xgb_model   = xgb_cached
            self.predictor.initialize_predictor(self.lstm_model)
            self.last_train_bars = len(features)
            self.feature_engineer._selected_columns   = cols_cached
            self.feature_engineer._fitted_scaler       = scaler_cached
            self.feature_engineer._fitted_close_scaler = close_scaler_cached
            # features is already aligned to cols_cached (T4 was skipped above)
            scaler      = scaler_cached
            close_scaler = close_scaler_cached
            self.logger.info(f"[ModelCache] Startup training skipped for {symbol} -- using cached models")
        else:
            # No cache or stale -- clear any partial injection and train fresh
            self.feature_engineer._selected_columns   = None
            self.feature_engineer._fitted_scaler       = None
            self.feature_engineer._fitted_close_scaler = None
            features, scaler, close_scaler, adjusted_lookback, direction = \
                self.feature_engineer.prepare_features(data, lookback, symbol_type)
            if features is None:
                self.logger.error(f"Failed to prepare features for fresh training for {symbol}, exiting")
                return
            if not await self.train_models(symbol, features, scaler, adjusted_lookback,
                                           direction=direction,
                                           selected_columns=self.feature_engineer._selected_columns):
                self.logger.error(f"Initial model training failed for {symbol}, exiting")
                return

        self.last_rmse = float('inf')

        # --- Recover any position that survived a crash/restart ---
        async with mt5_lock:
            existing = mt5.positions_get(symbol=symbol)
        if existing:
            # Collect ALL tickets for this symbol with our magic number
            _my_existing = [p for p in existing if getattr(p, 'magic', 0) == self.magic_number]
            if _my_existing:
                pos_type     = 'buy' if _my_existing[0].type == mt5.POSITION_TYPE_BUY else 'sell'
                _total_vol   = sum(p.volume for p in _my_existing)
                _total_pnl   = sum(getattr(p, 'profit', 0.0) for p in _my_existing)
                _n_tickets   = len(_my_existing)
                trading_agent.position = {
                    'type':           pos_type,
                    'entry':          _my_existing[0].price_open,
                    'volume':         _total_vol,
                    'partial_closed': _n_tickets > 1,   # partial already separated
                }
                trade_open_bar = 0
                self.logger.info(
                    f"[Recovery] Resumed {pos_type} position for {symbol}: "
                    f"{_n_tickets} ticket(s), total_vol={_total_vol}, "
                    f"tickets={[p.ticket for p in _my_existing]}"
                )
                update_dashboard(symbol,
                    status='in_trade (recovered)',
                    position=f"{pos_type.upper()} @ {_my_existing[0].price_open:.5f} x{_n_tickets}",
                    pnl=f"{_total_pnl:+.2f}",
                    price=f"{_my_existing[0].price_current:.5f}"
                          if hasattr(_my_existing[0], 'price_current') else '-',
                )

        # -- main loop --------------------------------------------------------
        bar_count = 0
        _last_bar_time = None

        async def _tick_price_refresher():
            """Update live price and combined floating P&L every 3 seconds.
            Aggregates both main and partial TP tickets so dashboard shows true total."""
            while not _shutdown_event.is_set():
                try:
                    async with mt5_lock:
                        _t        = mt5.symbol_info_tick(symbol)
                        _pos_list = mt5.positions_get(symbol=symbol)
                    if _t:
                        _mid    = (_t.bid + _t.ask) / 2
                        _kwargs = {'price': f"{_mid:.5f}"}
                        # Collect ALL magic-matched tickets (main + partial TP)
                        _my = [_p for _p in (_pos_list or [])
                               if getattr(_p, 'magic', 0) == self.magic_number]
                        if _my:
                            _dir         = 'BUY' if _my[0].type == mt5.POSITION_TYPE_BUY else 'SELL'
                            _total_vol   = sum(p.volume for p in _my)
                            _total_pnl   = sum(p.profit for p in _my)
                            _n           = len(_my)
                            _kwargs['position'] = f"{_dir} @ {_my[0].price_open:.5f} x{_n}"
                            _kwargs['pnl']      = f"{_total_pnl:+.2f}"
                            _kwargs['status']   = 'in_trade'
                        update_dashboard(symbol, **_kwargs)
                except Exception:
                    pass
                await asyncio.sleep(3)

        asyncio.create_task(_tick_price_refresher())
        # Sentinel: if a position was recovered, start S3 timer from bar 0
        # so the bot gets a full max_bars_open window before the time-stop fires.
        if trade_open_bar is not None:
            trade_open_bar = 0
        while True:
            if not await self.market_checker.is_market_open(symbol):
                self.logger.info(f"Market closed for {symbol}, waiting 60 s")
                await asyncio.sleep(60)
                continue

            async with mt5_lock:
                account_info = mt5.account_info()
            account_equity  = account_info.equity  if account_info else 10000
            account_balance = account_info.balance if account_info else 10000

            limit_status = await self.risk_manager.check_limits(account_equity, account_balance)

            if limit_status == 'close_all':
                floating_profit = account_equity - account_balance
                self.logger.info(
                    f"[AccountTP] Equity profit threshold hit -- closing ALL account positions "
                    f"(equity={account_equity:.2f}, balance={account_balance:.2f}, "
                    f"floating={floating_profit:.2f})"
                )
                await self.send_alert(
                    f"[AccountTP] Equity TP hit: floating profit {floating_profit:.2f} -- "
                    f"closing all positions on account"
                )
                # Close every open position on the entire account -- no symbol or magic filter.
                async with mt5_lock:
                    all_positions = mt5.positions_get()
                closed, failed = 0, 0
                for pos in (all_positions or []):
                    pos_type_opposite = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY \
                                        else mt5.ORDER_TYPE_BUY
                    base_req = {
                        "action":    mt5.TRADE_ACTION_DEAL,
                        "symbol":    pos.symbol,
                        "volume":    pos.volume,
                        "type":      pos_type_opposite,
                        "position":  pos.ticket,
                        "magic":     pos.magic,
                        "type_time": mt5.ORDER_TIME_GTC,
                    }
                    result = None
                    async with mt5_lock:
                        for _, mode in [
                            ("IOC",    mt5.ORDER_FILLING_IOC),
                            ("FOK",    mt5.ORDER_FILLING_FOK),
                            ("RETURN", mt5.ORDER_FILLING_RETURN),
                        ]:
                            req = {**base_req, "type_filling": mode}
                            result = mt5.order_send(req)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                break
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.logger.info(
                            f"[AccountTP] Closed {pos.symbol} ticket={pos.ticket} "
                            f"vol={pos.volume} profit={pos.profit:.2f}"
                        )
                        closed += 1
                    else:
                        self.logger.error(
                            f"[AccountTP] Failed to close {pos.symbol} ticket={pos.ticket}: "
                            f"{result.comment if result else 'no response'}"
                        )
                        failed += 1
                self.logger.info(f"[AccountTP] Sweep complete: {closed} closed, {failed} failed")
                trading_agent.save_q_table()
                break

            if limit_status == 'stop':
                self.logger.error(f"Risk/profit limit reached for {symbol} -- stopping")
                await self.send_alert(f"Risk/profit limit reached for {symbol}")
                trading_agent.save_q_table()
                update_dashboard(symbol, status='STOPPED -- risk limit')
                break

            # --- refresh data + features ---
            data = await self.data_fetcher.fetch_data(symbol, bars=lookback + 5000)
            if data is None:
                self.logger.warning(f"Retrying data fetch for {symbol} in 15 s")
                await asyncio.sleep(15)
                continue
            features, new_scaler, close_scaler, adjusted_lookback, direction = \
                self.feature_engineer.prepare_features(data, lookback, symbol_type)
            if features is None:
                self.logger.warning(f"Retrying feature preparation for {symbol} in 15 s")
                await asyncio.sleep(15)
                continue
            # --- Trailing stop (every poll cycle, intra-bar) ---
            # Also detects external SL/TP closures so dashboard updates immediately
            async with mt5_lock:
                _trail_positions = mt5.positions_get(symbol=symbol)
            _magic_positions = [
                p for p in (_trail_positions or [])
                if p.symbol == symbol and getattr(p, 'magic', 0) == self.magic_number
            ]
            if _magic_positions:
                await self.trade_executor.update_trailing_stop(symbol)
            elif trading_agent.position is not None:
                # Position was in trading_agent but is no longer in MT5 --
                # closed externally by SL/TP/margin call. Reset state immediately.
                self.logger.info(
                    f"[SL/TP] {symbol} position closed externally (SL/TP/margin) -- "
                    f"resetting state"
                )
                trading_agent.position = None
                trade_open_bar = None
                last_trade_time = bar_count
                update_dashboard(symbol,
                    status='running',
                    position='-',
                    pnl='(SL/TP)',
                )

            latest_bar_time = features.index[-1]
            if _last_bar_time is not None and latest_bar_time <= _last_bar_time:
                # Still same candle -- sleep a fraction of the timeframe before re-polling
                _same_bar_sleep = {
                    mt5.TIMEFRAME_M1: 10, mt5.TIMEFRAME_M5: 20,
                    mt5.TIMEFRAME_M15: 45, mt5.TIMEFRAME_M30: 90,
                    mt5.TIMEFRAME_H1: 240, mt5.TIMEFRAME_H4: 480,
                    mt5.TIMEFRAME_D1: 1200,
                }.get(self.data_fetcher.timeframe, 45)
                await asyncio.sleep(_same_bar_sleep)
                continue
            _last_bar_time = latest_bar_time
            bar_count += 1
            self.logger.debug(f"[Bar] {symbol} new bar #{bar_count} at {latest_bar_time}")
            update_dashboard(symbol, status='running', bars=bar_count)

            # --- models guard ---
            if self.lstm_model is None or self.xgb_model is None:
                self.logger.error(f"Models not trained for {symbol}, skipping")
                await asyncio.sleep(15)
                continue

            # --- scheduled retrain ---
            if bar_count % retrain_interval == 0 and len(features) > adjusted_lookback + 50:
                self.logger.info(f"Scheduled retrain for {symbol} at bar {bar_count}")
                # Reset cache so T4 re-selects on the latest data
                self.feature_engineer._selected_columns   = None
                self.feature_engineer._fitted_scaler       = None
                self.feature_engineer._fitted_close_scaler = None
                rt_features, rt_scaler, _, _, rt_direction = \
                    self.feature_engineer.prepare_features(data, lookback, symbol_type)
                if rt_features is not None:
                    if await self.train_models(symbol, rt_features, rt_scaler, adjusted_lookback,
                                               direction=rt_direction,
                                               selected_columns=self.feature_engineer._selected_columns):
                        features, scaler, direction = rt_features, rt_scaler, rt_direction

            # --- current ATR: read from the main feature window, not a tiny re-fetch ---
            # A 100-bar window cannot reliably compute ADX/MACD (needs 50+ bars of warmup).
            # features already has 4000-5000 bars with all indicators computed correctly.
            try:
                atr = float(features['atr'].iloc[-1]) if 'atr' in features.columns else None
            except Exception:
                atr = None
            if atr is None or np.isnan(atr):
                # Fallback: compute ATR directly from the raw OHLC data
                atr_series = self.feature_engineer.calculate_atr(data)
                atr = float(atr_series.iloc[-1]) if (atr_series is not None and not atr_series.empty) else None

            # --- [Drift detection] via direction accuracy ---
            # Guards:
            #   1. Need at least 50 predictions for a statistically meaningful accuracy
            #   2. Cooldown of drift_retrain_cooldown bars between drift retrains
            #      to prevent back-to-back retrains on early noisy predictions
            drift_min_samples      = self.config.get('drift_min_samples', 50)
            drift_retrain_cooldown = self.config.get('drift_retrain_cooldown', 50)
            bars_since_drift       = bar_count - getattr(self, '_last_drift_bar', -drift_retrain_cooldown)

            if (len(self.predictor.recent_predictions) >= drift_min_samples
                    and bars_since_drift >= drift_retrain_cooldown):
                recent_preds = list(self.predictor.recent_predictions)[-drift_min_samples:]
                recent_acc = accuracy_score(
                    [p[0] for p in recent_preds],
                    [int(np.argmax(p[1])) for p in recent_preds]
                )
                drift_threshold = self.config.get('drift_acc_threshold', 0.40)
                if recent_acc < drift_threshold:
                    self.logger.warning(
                        f"[Drift] {symbol} accuracy={recent_acc:.3f} < {drift_threshold} "
                        f"over last {drift_min_samples} bars -- retraining"
                    )
                    await self.send_alert(f"Model drift for {symbol}: accuracy={recent_acc:.3f}")
                    self._last_drift_bar = bar_count
                    self.feature_engineer._selected_columns   = None
                    self.feature_engineer._fitted_scaler       = None
                    self.feature_engineer._fitted_close_scaler = None
                    dr_features, dr_scaler, _, _, dr_direction = \
                        self.feature_engineer.prepare_features(data, lookback, symbol_type)
                    if dr_features is not None:
                        if await self.train_models(symbol, dr_features, dr_scaler, adjusted_lookback,
                                                   direction=dr_direction,
                                                   use_tuner=self.config.get('tune_on_drift', False),
                                                   selected_columns=self.feature_engineer._selected_columns):
                            features, scaler, direction = dr_features, dr_scaler, dr_direction
                else:
                    self.logger.debug(f"[Drift] {symbol} accuracy={recent_acc:.3f} OK")

            current_price = features['close'].iloc[-1]
            # Use live tick price for dashboard (more accurate than last closed candle)
            try:
                async with mt5_lock:
                    _live_tick = mt5.symbol_info_tick(symbol)
                if _live_tick:
                    _live_price = (_live_tick.bid + _live_tick.ask) / 2
                    update_dashboard(symbol, price=f"{_live_price:.5f}")
                else:
                    update_dashboard(symbol, price=f"{current_price:.5f}")
            except Exception:
                update_dashboard(symbol, price=f"{current_price:.5f}")

            # --- manage open position ---
            async with mt5_lock:
                positions = mt5.positions_get(symbol=symbol)
            has_position = any(
                p.symbol == symbol and getattr(p, 'magic', 0) == self.magic_number
                for p in (positions or [])
            )
            # Update dashboard: aggregate ALL open tickets for this symbol (main + partial TP)
            if has_position and positions:
                _my_pos = [p for p in positions
                           if p.symbol == symbol and getattr(p, 'magic', 0) == self.magic_number]
                if _my_pos:
                    _dir        = 'BUY' if _my_pos[0].type == mt5.POSITION_TYPE_BUY else 'SELL'
                    _total_vol  = sum(p.volume for p in _my_pos)
                    _total_pnl  = sum(p.profit for p in _my_pos)
                    _entry      = _my_pos[0].price_open  # use first ticket's entry
                    _n_tickets  = len(_my_pos)
                    update_dashboard(symbol,
                        status='in_trade',
                        position=f"{_dir} @ {_entry:.5f} x{_n_tickets}",
                        pnl=f"{_total_pnl:+.2f}",
                        entry=f"{_entry:.5f}",
                    )

            if has_position:
                # Trailing stop runs every poll cycle in the block above (intra-bar).
                # Derive pos_type from trading_agent.position on every iteration so it is
                # always defined -- relying on the pre-loop recovery assignment caused an
                # UnboundLocalError on a fresh start with no pre-existing position.
                pos_type = trading_agent.position['type'] if trading_agent.position else None

                predicted_action, confidence = self.predictor.predict(
                    self.lstm_model, self.xgb_model, features, scaler, close_scaler,
                    list(features.columns), current_price, symbol_type, atr=atr, symbol=symbol
                )
                bars_open = (bar_count - trade_open_bar) if trade_open_bar is not None else 0

                should_close = False
                close_reason = ""
                # [S2] ATR-normalised directional reversal exit
                if pos_type == 'buy'  and predicted_action == 'sell':
                    should_close = True; close_reason = "direction reversal (sell while long)"
                if pos_type == 'sell' and predicted_action == 'buy':
                    should_close = True; close_reason = "direction reversal (buy while short)"
                # [S3] Max-bars-open time stop
                if bars_open >= max_bars_open:
                    should_close = True; close_reason = f"max bars open ({bars_open}/{max_bars_open})"

                if should_close:
                    self.logger.info(f"Closing {symbol}: {close_reason}")
                    entry = trading_agent.position['entry'] if trading_agent.position else current_price
                    vol   = trading_agent.position['volume'] if trading_agent.position else 0
                    close_ok, closed_tickets, open_timestamps = await self.trade_executor.close_position(symbol, trading_agent)

                    # Fetch actual MT5 P&L summed across all tickets (main + partial TP)
                    actual_pnl = None
                    if close_ok and closed_tickets:
                        actual_pnl = await self.trade_executor.get_actual_pnl(
                            symbol, closed_tickets, open_timestamps
                        )

                    if actual_pnl is not None:
                        reward = actual_pnl
                        self.logger.info(f"[Journal] Actual MT5 P&L for {symbol}: {actual_pnl:.4f}")
                    else:
                        # Fallback: synthetic reward (used for Q-table only)
                        reward = await trading_agent.calculate_reward(
                            pos_type, entry, current_price, vol,
                            predicted_action=predicted_action
                        )
                        self.logger.debug(f"[Journal] Synthetic P&L fallback for {symbol}: {reward:.4f}")

                    state = trading_agent.discretize_state(features)
                    trading_agent.update_q_table(state, 'close', reward, state)
                    trading_agent.save_q_table()
                    self.risk_manager.update_history(reward)
                    last_trade_time = bar_count
                    trade_open_bar  = None
                    update_dashboard(symbol,
                        status='running',
                        position='-',
                        pnl=f"{reward:+.2f}",
                    )

                    # Improvement 10: write journal row with actual P&L
                    try:
                        journal.log(
                            symbol=symbol, direction=pos_type,
                            entry=entry, exit_price=current_price,
                            sl=_entry_sl, tp=_entry_tp,
                            volume=vol, pnl=reward,
                            confidence=_entry_confidence,
                            adx_at_entry=_entry_adx,
                            reason=close_reason,
                            bars_held=bars_open,
                        )
                    except Exception:
                        pass

                    # [P3] Feed actual outcome into predictor so ensemble weights can learn
                    # Map pos_type to direction class: buy=2 (up), sell=0 (down)
                    true_class = 2 if pos_type == 'buy' else 0
                    if self.predictor.recent_predictions:
                        last = self.predictor.recent_predictions[-1]
                        self.predictor.record_outcome(
                            true_class,
                            last[1],   # lstm_probs from last prediction
                            last[2],   # xgb_class
                            last[3],   # ensemble_class
                        )

                await asyncio.sleep(15)
                continue

            # --- [E4] ATR-velocity cooldown ---
            cooldown_min = self.config.get('cooldown_min_bars', 1)
            cooldown_max = self.config.get('cooldown_max_bars', 10)
            if atr is not None and 'high' in features.columns and 'low' in features.columns:
                avg_range     = (features['high'] - features['low']).iloc[-20:].mean()
                cooldown_bars = int(np.clip(atr / (avg_range + 1e-10), cooldown_min, cooldown_max))
            else:
                cooldown_bars = cooldown_min
            if (bar_count - last_trade_time) < cooldown_bars:
                if self.config.get('log_filter_blocks', True):
                    self.logger.info(
                        f"[E4/Cooldown] {symbol}: {bar_count - last_trade_time}/{cooldown_bars} bars "
                        f"since last trade"
                    )
                await asyncio.sleep(_same_bar_sleep if '_same_bar_sleep' in locals() else 45)
                continue

            # --- Improvement 1: spread filter ---
            # Skip entry when current spread > spread_max_multiplier * rolling avg spread
            spread_max_mult = self.config.get('spread_max_multiplier', 2.5, symbol=symbol)
            try:
                async with mt5_lock:
                    _tick = mt5.symbol_info_tick(symbol)
                if _tick is not None:
                    _sym_info = mt5.symbol_info(symbol)
                    _point    = _sym_info.point if _sym_info else 1e-5
                    cur_spread = (_tick.ask - _tick.bid) / _point
                    avg_spread = self.data_fetcher._avg_spread.get(symbol, cur_spread)
                    # Update rolling average (exponential, alpha=0.05)
                    self.data_fetcher._avg_spread[symbol] = 0.95 * avg_spread + 0.05 * cur_spread
                    if avg_spread > 0 and cur_spread > spread_max_mult * avg_spread:
                        if self.config.get('log_filter_blocks', True):
                            self.logger.info(
                                f"[SpreadFilter] {symbol} spread={cur_spread:.1f} pts > "
                                f"{spread_max_mult}x avg={avg_spread:.1f} pts -- skipping entry"
                            )
                        await asyncio.sleep(_same_bar_sleep if '_same_bar_sleep' in locals() else 45)
                        continue
            except Exception as _e:
                self.logger.debug(f"[SpreadFilter] {symbol}: {_e}")

            # --- Improvement 3: session gate ---
            # Only block the true dead-of-night window (minimal liquidity, widest spreads).
            # The old 22:00-07:00 block was too wide -- London pre-open (06:00-07:00 UTC)
            # and late US session (22:00-23:00 UTC) still have reasonable liquidity.
            # Configurable: session_block_start / session_block_end (UTC hours).
            _now_utc = datetime.now(timezone.utc)
            _utc_h   = _now_utc.hour
            _skip_session = False
            session_gate = self.config.get('session_gate', True)
            if session_gate:
                if symbol_type == 'forex':
                    block_start = self.config.get('session_block_start', 0)   # 00:00 UTC
                    block_end   = self.config.get('session_block_end',   5)   # 05:00 UTC
                    if block_start <= _utc_h < block_end:
                        _skip_session = True
                elif symbol_type == 'crypto':
                    # Crypto: only block weekend pre-dawn (very low volume)
                    if _now_utc.weekday() >= 5 and _utc_h < 4:
                        _skip_session = True
            if _skip_session:
                self.logger.debug(f"[SessionGate] {symbol} -- outside liquid window ({_utc_h:02d}:00 UTC), skip entry")
                await asyncio.sleep(_same_bar_sleep if '_same_bar_sleep' in locals() else 45)
                continue

            # --- predict direction ---
            predicted_action, confidence = self.predictor.predict(
                self.lstm_model, self.xgb_model, features, scaler, close_scaler,
                list(features.columns), current_price, symbol_type, atr=atr, symbol=symbol
            )
            update_dashboard(symbol,
                action=predicted_action,
                confidence=f"{confidence:.3f}"
            )
            # Feed last prediction into deque for drift detection accuracy tracking.
            # We append only the raw prediction components here (no true label yet --
            # that is only known at close). The deque is used for accuracy_score in
            # drift detection, which compares recent_predictions[i][0] (true class)
            # against argmax(recent_predictions[i][1]) (lstm class). Since we don't
            # know the true class at entry, we defer this entirely to record_outcome
            # at close, where the actual market direction is known.
            # (Provisional appends were removed -- they biased ensemble weight learning)

            # [C1][C2] Confidence gate -- no floor, truly blocks low-quality signals
            if not self.predictor.confidence_ok(confidence, min_confidence):
                if self.config.get('log_filter_blocks', True):
                    self.logger.info(
                        f"[ConfGate] {symbol}: conf={confidence:.3f} < {min_confidence} "
                        f"(action={predicted_action}) -- skip"
                    )
                await asyncio.sleep(_same_bar_sleep if '_same_bar_sleep' in locals() else 45)
                continue

            # --- Improvement 5: multi-timeframe confirmation ---
            # Skip MTF for symbols already running on H4 or D1 -- they ARE the higher TF.
            _is_h4_or_higher = self.data_fetcher.timeframe in (
                mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1
            )
            mtf_enabled = self.config.get('mtf_confirmation', True) and not _is_h4_or_higher
            if mtf_enabled and symbol_type in ('forex', 'crypto') and predicted_action != 'hold':
                try:
                    _h4_data = await self._h4_fetcher.fetch_data(symbol, bars=120)
                    if _h4_data is not None and len(_h4_data) >= 60:
                        _h4_ema20 = _h4_data['close'].ewm(span=20, adjust=False).mean().iloc[-1]
                        _h4_sma50 = _h4_data['close'].rolling(50).mean().iloc[-1]
                        h4_up   = _h4_ema20 > _h4_sma50
                        h4_down = _h4_ema20 < _h4_sma50
                        if predicted_action == 'buy'  and not h4_up:
                            self.logger.info(
                                f"[MTF] {symbol} H4 trend DOWN "
                                f"(EMA20={_h4_ema20:.5f} < SMA50={_h4_sma50:.5f}) -- rejecting BUY"
                            )
                            await asyncio.sleep(_same_bar_sleep if '_same_bar_sleep' in dir() else 45)
                            continue
                        if predicted_action == 'sell' and not h4_down:
                            self.logger.info(
                                f"[MTF] {symbol} H4 trend UP "
                                f"(EMA20={_h4_ema20:.5f} > SMA50={_h4_sma50:.5f}) -- rejecting SELL"
                            )
                            await asyncio.sleep(_same_bar_sleep if '_same_bar_sleep' in dir() else 45)
                            continue
                except Exception as _e:
                    self.logger.warning(f"[MTF] {symbol} H4 fetch failed: {_e} -- skipping MTF check")

            # --- Q-table + [E1] trend + [E2] ADX filter ---
            state  = trading_agent.discretize_state(features)
            trading_agent._last_confidence = confidence   # used by confidence override in choose_action
            action = trading_agent.choose_action(state, predicted_action, features)

            if action == 'hold':
                await asyncio.sleep(15)
                continue

            volume = await self.risk_manager.calculate_lot_size(
                symbol, account_equity, symbol_type, current_price, atr or 0
            )

            # --- Improvement 7: profit-factor position scaling ---
            # If recent closed trades show profit factor > 1.5, scale up by up to 1.3x.
            # If profit factor < 0.8 (losing streak), scale down to 0.7x.
            try:
                pf_history = list(self.risk_manager.history)
                if len(pf_history) >= 10:
                    gross_profit = sum(p for p in pf_history if p > 0) or 1e-6
                    gross_loss   = abs(sum(p for p in pf_history if p < 0)) or 1e-6
                    profit_factor = gross_profit / gross_loss
                    if profit_factor > 1.5:
                        pf_scale = min(1.3, 0.9 + 0.1 * profit_factor)
                    elif profit_factor < 0.8:
                        pf_scale = max(0.7, profit_factor)
                    else:
                        pf_scale = 1.0
                    volume = volume * pf_scale
                    self.logger.info(f"[ProfitFactor] {symbol} PF={profit_factor:.2f} -> scale={pf_scale:.2f}, vol={volume:.4f}")
            except Exception as _e:
                self.logger.debug(f"[ProfitFactor] {symbol}: {_e}")

            # --- Improvement 9: correlation guard (multi-symbol) ---
            # Don't open a same-direction trade if a highly correlated symbol is already open.
            # Correlation pairs are configurable; default: EURUSD/GBPUSD, BTC/ETH, etc.
            corr_guard_enabled = self.config.get('correlation_guard', True)
            if corr_guard_enabled and predicted_action != 'hold':
                corr_pairs = self.config.get('corr_pairs', {
                    'EURUSD': ['GBPUSD', 'EURUSD+'], 'EURUSD+': ['GBPUSD', 'EURUSD'],
                    'GBPUSD': ['EURUSD', 'EURUSD+'],
                    'BTCUSD': ['ETHUSD', 'BTCUSD+', 'ETHUSD+'], 'BTCUSD+': ['ETHUSD', 'ETHUSD+'],
                    'ETHUSD': ['BTCUSD', 'BTCUSD+', 'ETHUSD+'], 'ETHUSD+': ['BTCUSD', 'BTCUSD+'],
                    'XAUUSD': ['XAUUSD+'], 'XAUUSD+': ['XAUUSD'],
                })
                corr_blocked = False
                for corr_sym in corr_pairs.get(symbol, []):
                    async with mt5_lock:
                        corr_pos = mt5.positions_get(symbol=corr_sym)
                    if corr_pos:
                        for cp in corr_pos:
                            cp_dir = 'buy' if cp.type == mt5.POSITION_TYPE_BUY else 'sell'
                            if cp_dir == predicted_action:
                                self.logger.info(
                                    f"[CorrGuard] Blocking {predicted_action} on {symbol}: "
                                    f"correlated {corr_sym} already {cp_dir}"
                                )
                                corr_blocked = True
                                break
                    if corr_blocked:
                        break
                if corr_blocked:
                    await asyncio.sleep(15)
                    continue

            success, deal, sl, action_taken = await self.trade_executor.execute_trade(
                symbol, action, volume, predicted_action, current_price,
                1.0, 1.0, trading_agent, symbol_type, atr=atr
            )

            if success and action_taken not in (None, 'hold'):
                trade_open_bar    = bar_count
                _entry_confidence = confidence
                _entry_adx        = float(features['adx'].iloc[-1]) if 'adx' in features.columns else 0.0
                _entry_sl         = sl or 0.0
                async with mt5_lock:
                    _pos = mt5.positions_get(symbol=symbol)
                # Collect all open tickets for this symbol (main + partial TP)
                _my_new = [p for p in (_pos or []) if getattr(p, 'magic', 0) == self.magic_number]
                _entry_tp   = float(_my_new[0].tp) if _my_new else 0.0
                _n_tickets  = len(_my_new)
                _total_vol  = sum(p.volume for p in _my_new) if _my_new else volume
                _today_trades = _dashboard_state.get(symbol, {}).get('trades_today', 0) + 1
                update_dashboard(symbol,
                    status='in_trade',
                    position=f"{action_taken.upper()} @ {current_price:.5f} x{_n_tickets}",
                    trades_today=_today_trades,
                )
                # Store combined volume so journal/reward use the true total across both tickets
                if trading_agent.position is not None:
                    trading_agent.position['volume'] = _total_vol
                trading_agent.update_q_table(state, action, 0.0, trading_agent.discretize_state(features))
                trading_agent.save_q_table()
                last_trade_time = bar_count
                await self.send_alert(f"Trade: {symbol} {action} @ {current_price:.5f}")
                # NOTE: trailing-stop updates happen naturally in the has_position block
                # at the top of each bar iteration -- no separate blocking loop needed.

            # Sleep for a fraction of the timeframe before polling again.
            _tf_poll_seconds = {
                mt5.TIMEFRAME_M1:  15, mt5.TIMEFRAME_M5:  30,
                mt5.TIMEFRAME_M15: 60, mt5.TIMEFRAME_M30: 120,
                mt5.TIMEFRAME_H1: 300, mt5.TIMEFRAME_H4: 600,
                mt5.TIMEFRAME_D1: 1800,
            }.get(self.data_fetcher.timeframe, 60)
            await asyncio.sleep(_tf_poll_seconds)

# ---------------------------------------------------------------------------
# Per-symbol thread runner
# ---------------------------------------------------------------------------
_shutdown_event   = threading.Event()   # global: shut down ALL threads
_symbol_stop_events: dict = {}          # symbol -> threading.Event: stop ONE symbol

def run_symbol_in_thread(sym: str, backtest: bool):
    """Each symbol runs in its own OS thread with its own asyncio event loop.
    Respects both _shutdown_event (global) and _symbol_stop_events[sym] (per-symbol).
    """
    stop_ev = threading.Event()
    _symbol_stop_events[sym] = stop_ev

    update_dashboard(sym, status='starting')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        bot = TradingBot()
        bot_instances.append(bot)

        async def _run_with_shutdown():
            run_task = asyncio.create_task(bot.run(sym, backtest))
            while not run_task.done():
                if _shutdown_event.is_set() or stop_ev.is_set():
                    run_task.cancel()
                    try:
                        await run_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        _dummy_agent = TradingAgent(
                            bot.config, bot.logger, sym, sym.replace('+', ''), 'forex'
                        )
                        await bot.trade_executor.close_position(sym, _dummy_agent)
                    except Exception:
                        pass
                    reason = 'shutdown' if _shutdown_event.is_set() else 'removed'
                    update_dashboard(sym, status=reason)
                    return
                await asyncio.sleep(1)
            return await run_task

        loop.run_until_complete(_run_with_shutdown())
        update_dashboard(sym, status='finished')
    except Exception as e:
        update_dashboard(sym, status='ERROR', error=str(e)[:60])
        logging.getLogger('trading_bot').error(f"{sym} thread error: {e}", exc_info=True)
    finally:
        _symbol_stop_events.pop(sym, None)
        loop.close()


# ---------------------------------------------------------------------------
# HTTP Dashboard server (runs in its own daemon thread)
# ---------------------------------------------------------------------------
def _build_dashboard_html() -> str:
    """Generate the full dashboard HTML page from current _dashboard_state."""
    with _dashboard_lock:
        state_copy    = {k: dict(v) for k, v in _dashboard_state.items()}
        history_copy  = {k: list(v) for k, v in _pnl_history.items()}

    # Compute account-level totals
    total_pnl      = 0.0
    open_positions = 0
    trades_today   = 0
    for s in state_copy.values():
        try:
            pv = float(str(s.get('pnl','')).replace('+','').replace('(SL/TP)','0'))
            total_pnl += pv
        except Exception:
            pass
        if 'in_trade' in s.get('status', ''):
            open_positions += 1
        trades_today += int(s.get('trades_today', 0))

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Build symbol rows
    rows = []
    for sym in sorted(state_copy.keys()):
        s       = state_copy[sym]
        status  = s.get('status', '-')
        price   = s.get('price', '-')
        pos     = s.get('position', '-')
        pnl     = s.get('pnl', '-')
        conf    = s.get('confidence', '-')
        action  = s.get('action', '-')
        bars    = s.get('bars', 0)
        trades  = s.get('trades_today', 0)
        updated = s.get('last_update', '-')
        error   = s.get('error', '')

        # Status badge colour
        if 'ERROR' in status.upper():
            badge = f'<span class="badge badge-error">{status}</span>'
        elif 'in_trade' in status.lower():
            badge = f'<span class="badge badge-trade">IN TRADE</span>'
        elif 'running' in status.lower():
            badge = f'<span class="badge badge-ok">running</span>'
        elif 'starting' in status.lower():
            badge = f'<span class="badge badge-warn">starting</span>'
        elif 'removed' in status.lower() or 'shutdown' in status.lower() or 'finished' in status.lower():
            badge = f'<span class="badge badge-off">{status}</span>'
        else:
            badge = f'<span class="badge">{status}</span>'

        # Position colour
        if 'BUY' in pos:
            pos_html = f'<span class="buy">{pos}</span>'
        elif 'SELL' in pos:
            pos_html = f'<span class="sell">{pos}</span>'
        else:
            pos_html = f'<span class="muted">{pos}</span>'

        # P&L colour
        try:
            pnl_f    = float(str(pnl).replace('+','').replace('(SL/TP)','0'))
            pnl_cls  = 'profit' if pnl_f >= 0 else 'loss'
            pnl_html = f'<span class="{pnl_cls}">{pnl}</span>'
        except Exception:
            pnl_html = f'<span class="muted">{pnl}</span>'

        # Signal arrow
        if action == 'buy':
            sig_html = '<span class="buy sig">&#9650; BUY</span>'
        elif action == 'sell':
            sig_html = '<span class="sell sig">&#9660; SELL</span>'
        else:
            sig_html = f'<span class="muted">{action}</span>'

        # Confidence bar
        try:
            conf_f   = float(conf)
            conf_pct = int(conf_f * 100)
            conf_cls = 'conf-high' if conf_f >= 0.8 else ('conf-mid' if conf_f >= 0.5 else 'conf-low')
            conf_html = (f'<div class="conf-bar-wrap" title="{conf}">'
                         f'<div class="conf-bar {conf_cls}" style="width:{conf_pct}%"></div>'
                         f'</div><small>{conf}</small>')
        except Exception:
            conf_html = f'<span class="muted">{conf}</span>'

        err_html = f'<div class="error-msg">{error}</div>' if error else ''

        rows.append(f"""
        <tr class="{'row-trade' if 'in_trade' in status.lower() else ''}">
          <td class="sym">{sym}</td>
          <td>{badge}{err_html}</td>
          <td class="mono">{price}</td>
          <td>{pos_html}</td>
          <td>{pnl_html}</td>
          <td>{conf_html}</td>
          <td>{sig_html}</td>
          <td class="center">{trades}</td>
          <td class="center">{bars}</td>
          <td class="muted mono small">{updated}</td>
        </tr>""")

    rows_html = '\n'.join(rows)

    # Sparkline data for JS
    spark_data = json.dumps({
        sym: [[round(t, 0), v] for t, v in hist]
        for sym, hist in history_copy.items()
    })

    total_cls  = 'profit' if total_pnl >= 0 else 'loss'
    total_sign = '+' if total_pnl >= 0 else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="3">
  <title>Trading Bot Dashboard</title>
  <style>
    :root {{
      --bg:       #0d1117;
      --panel:    #161b22;
      --border:   #30363d;
      --text:     #e6edf3;
      --muted:    #8b949e;
      --buy:      #3fb950;
      --sell:     #f85149;
      --warn:     #d29922;
      --blue:     #58a6ff;
      --purple:   #bc8cff;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg); color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px;
      padding: 16px;
    }}
    h1 {{ font-size: 20px; color: var(--blue); margin-bottom: 4px; }}
    .subtitle {{ color: var(--muted); font-size: 12px; margin-bottom: 16px; }}
    /* Summary cards */
    .cards {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
    .card {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 12px 20px; min-width: 140px;
    }}
    .card-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}
    .card-value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    /* Main table */
    .tbl-wrap {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; overflow: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead th {{
      background: var(--bg); color: var(--muted); font-size: 11px;
      text-transform: uppercase; letter-spacing: .5px;
      padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border);
    }}
    tbody tr {{ border-bottom: 1px solid var(--border); transition: background .15s; }}
    tbody tr:hover {{ background: #1c2128; }}
    tbody tr.row-trade {{ background: #0d1f17; }}
    td {{ padding: 10px 12px; vertical-align: middle; }}
    .sym {{ font-weight: 700; color: var(--blue); font-size: 13px; }}
    .mono {{ font-family: 'Consolas', monospace; }}
    .small {{ font-size: 11px; }}
    .muted {{ color: var(--muted); }}
    .center {{ text-align: center; }}
    .buy  {{ color: var(--buy); font-weight: 600; }}
    .sell {{ color: var(--sell); font-weight: 600; }}
    .profit {{ color: var(--buy); font-weight: 600; }}
    .loss   {{ color: var(--sell); font-weight: 600; }}
    .sig {{ font-size: 12px; }}
    /* Badges */
    .badge {{
      display: inline-block; padding: 2px 8px; border-radius: 12px;
      font-size: 11px; font-weight: 600; letter-spacing: .3px;
      background: var(--border); color: var(--muted);
    }}
    .badge-ok    {{ background: #1a3a1e; color: var(--buy); }}
    .badge-trade {{ background: #0d3320; color: #4ade80; border: 1px solid var(--buy); }}
    .badge-warn  {{ background: #3a2a0a; color: var(--warn); }}
    .badge-error {{ background: #3a0d0d; color: var(--sell); }}
    .badge-off   {{ background: var(--border); color: var(--muted); }}
    /* Confidence bar */
    .conf-bar-wrap {{
      background: var(--border); border-radius: 4px;
      height: 4px; width: 80px; margin-bottom: 3px; overflow: hidden;
    }}
    .conf-bar {{ height: 100%; border-radius: 4px; }}
    .conf-high {{ background: var(--buy); }}
    .conf-mid  {{ background: var(--warn); }}
    .conf-low  {{ background: var(--sell); }}
    .error-msg {{ color: var(--sell); font-size: 10px; margin-top: 2px; }}
    /* Sparkline canvas */
    .spark {{ display: block; margin-top: 8px; }}
    /* Footer */
    .footer {{
      margin-top: 16px; color: var(--muted); font-size: 11px; text-align: right;
    }}
    /* Responsive */
    @media (max-width: 900px) {{
      .card-value {{ font-size: 18px; }}
    }}
  </style>
</head>
<body>
  <h1>&#9678; Trading Bot Dashboard</h1>
  <div class="subtitle">Auto-refreshes every 3 seconds &nbsp;|&nbsp; {now_str}</div>

  <div class="cards">
    <div class="card">
      <div class="card-label">Symbols</div>
      <div class="card-value" style="color:var(--blue)">{len(state_copy)}</div>
    </div>
    <div class="card">
      <div class="card-label">Open Positions</div>
      <div class="card-value" style="color:var(--buy)">{open_positions}</div>
    </div>
    <div class="card">
      <div class="card-label">Trades Today</div>
      <div class="card-value" style="color:var(--purple)">{trades_today}</div>
    </div>
    <div class="card">
      <div class="card-label">Session P&amp;L</div>
      <div class="card-value {total_cls}">{total_sign}{total_pnl:.2f}</div>
    </div>
  </div>

  <div class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>Status</th><th>Live Price</th>
          <th>Position</th><th>P&amp;L</th><th>Confidence</th>
          <th>Signal</th><th>Trades</th><th>Bars</th><th>Updated</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>

  <!-- P&L sparkline chart -->
  <div style="margin-top:20px;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px">
    <div style="color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px">Session P&amp;L History</div>
    <canvas id="sparkCanvas" height="80" style="width:100%;display:block"></canvas>
  </div>

  <div class="footer">
    Trading Bot v14 &nbsp;|&nbsp;
    <a href="/api/state" style="color:var(--blue);text-decoration:none">JSON API</a>
  </div>

  <script>
  // Draw sparklines from P&L history
  const data = {spark_data};
  const canvas = document.getElementById('sparkCanvas');
  canvas.width = canvas.offsetWidth || 800;
  const ctx = canvas.getContext('2d');
  const colors = ['#58a6ff','#3fb950','#f85149','#bc8cff','#d29922',
                  '#79c0ff','#56d364','#ffa198','#d2a8ff','#e3b341'];
  let ci = 0;
  const allVals = Object.values(data).flat().map(d=>d[1]);
  const minV = allVals.length ? Math.min(...allVals) : -1;
  const maxV = allVals.length ? Math.max(...allVals) :  1;
  const range = maxV - minV || 1;
  const W = canvas.width, H = canvas.height, PAD = 6;

  // Zero line
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  const zeroY = H - PAD - ((0 - minV) / range) * (H - 2*PAD);
  ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(W, zeroY); ctx.stroke();

  for (const [sym, pts] of Object.entries(data)) {{
    if (!pts.length) continue;
    const tMin = pts[0][0], tMax = pts[pts.length-1][0] || tMin+1;
    const tRange = tMax - tMin || 1;
    ctx.strokeStyle = colors[ci++ % colors.length];
    ctx.lineWidth = 1.5; ctx.beginPath();
    pts.forEach(([t, v], i) => {{
      const x = PAD + ((t - tMin) / tRange) * (W - 2*PAD);
      const y = H - PAD - ((v - minV) / range) * (H - 2*PAD);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}
  </script>
</body>
</html>"""


def run_http_dashboard(stop_event: threading.Event, port: int = 8080):
    """Serve the dashboard on http://localhost:<port>
    Also exposes /api/state as a JSON endpoint for external tools.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # suppress access log noise

        def do_GET(self):
            if self.path in ('/', '/dashboard'):
                html = _build_dashboard_html().encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            elif self.path == '/api/state':
                with _dashboard_lock:
                    payload = json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'symbols':   {k: dict(v) for k, v in _dashboard_state.items()},
                    }, indent=2).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(('0.0.0.0', port), _Handler)
    server.timeout = 1.0   # allow checking stop_event every second
    logger = logging.getLogger('trading_bot')
    logger.info(f"[Dashboard] HTTP server started at http://localhost:{port}")
    logger.info(f"[Dashboard] Access from LAN at http://<your-ip>:{port}")
    while not stop_event.is_set():
        server.handle_request()
    server.server_close()
    logger.info("[Dashboard] HTTP server stopped.")


def run_dashboard(symbols: list, stop_event: threading.Event,
                  refresh_secs: float = 2.0, http_port: int = 8080):
    """Runs both:
      - HTTP dashboard on http://localhost:<http_port>  (primary)
      - Rich terminal table (fallback / secondary)
    """
    # Start HTTP server in its own sub-thread
    _http_thread = threading.Thread(
        target=run_http_dashboard,
        args=(stop_event, http_port),
        daemon=True, name="http-dashboard",
    )
    _http_thread.start()

    # Also keep rich terminal table running alongside
    try:
        from rich.live import Live
        from rich.table import Table
        from rich.console import Console
        from rich import box

        console = Console()

        def make_table():
            t = Table(
                title=f"Trading Bot -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  http://localhost:{http_port}",
                box=box.SIMPLE_HEAVY, show_lines=True, highlight=True,
            )
            t.add_column("Symbol",     style="cyan",    min_width=12)
            t.add_column("Status",     style="white",   min_width=16)
            t.add_column("Price",      style="yellow",  min_width=12, justify="right")
            t.add_column("Position",   style="green",   min_width=20)
            t.add_column("P&L",        style="magenta", min_width=10, justify="right")
            t.add_column("Conf",       style="blue",    min_width=8,  justify="right")
            t.add_column("Signal",     style="white",   min_width=8)
            t.add_column("Updated",    style="dim",     min_width=10)

            with _dashboard_lock:
                current_syms = sorted(_dashboard_state.keys())
                for sym in current_syms:
                    s       = _dashboard_state.get(sym, {})
                    status  = s.get('status', 'waiting')
                    pos     = s.get('position', '-')
                    pnl     = s.get('pnl', '-')
                    if 'ERROR' in status.upper():
                        st = f"[red]{status}[/red]"
                    elif 'in_trade' in status.lower():
                        st = "[green]IN TRADE[/green]"
                    elif status == 'running':
                        st = "[green]running[/green]"
                    else:
                        st = status
                    pos_s = f"[green]{pos}[/green]" if 'BUY' in pos else \
                            (f"[red]{pos}[/red]" if 'SELL' in pos else pos)
                    try:
                        pv = float(str(pnl).replace('+','').replace('(SL/TP)','0'))
                        pnl_s = f"[green]{pnl}[/green]" if pv >= 0 else f"[red]{pnl}[/red]"
                    except Exception:
                        pnl_s = pnl
                    sig = s.get('action', '-')
                    sig_s = f"[green]BUY[/green]" if sig == 'buy' else \
                            (f"[red]SELL[/red]" if sig == 'sell' else sig)
                    t.add_row(
                        sym, st,
                        s.get('price', '-'), pos_s, pnl_s,
                        s.get('confidence', '-'), sig_s,
                        s.get('last_update', '-'),
                    )
            return t

        with Live(make_table(), console=console, refresh_per_second=1/refresh_secs,
                  screen=False) as live:
            while not stop_event.is_set():
                live.update(make_table())
                time.sleep(refresh_secs)

    except ImportError:
        logger = logging.getLogger('trading_bot')
        logger.info("'rich' not installed -- terminal dashboard disabled. "
                    f"Use browser at http://localhost:{http_port}")
        while not stop_event.is_set():
            time.sleep(5)


# ---------------------------------------------------------------------------
# Multi-symbol concurrent execution entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Bot v14 Multi-Symbol")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EU50.r","JPN225ft","FRA40.r","HK50.r","XAUUSD+","EURUSD+",
                 "USDJPY+","SP500.r","DJ30.r","UK100.r","TSLA","NVIDIA"],
        help="Symbols to trade concurrently"
    )
    parser.add_argument("--backtest",          action="store_true", help="Run backtest mode")
    parser.add_argument("--no-dashboard",      action="store_true", help="Disable live dashboard")
    parser.add_argument("--dashboard-refresh", type=float, default=2.0,
                        help="Dashboard refresh interval in seconds (default 2)")
    parser.add_argument("--dashboard-port",    type=int,   default=8080,
                        help="HTTP dashboard port (default 8080)")
    args = parser.parse_args()

    bot_instances    = []
    _stop_dashboard  = threading.Event()  # created before signal handler so it's in scope

    # Register shutdown signals in the main thread (only place Python allows it)
    def _main_shutdown(sig, frame):
        _init_logger.info(f"Shutdown signal {sig} received -- stopping all symbol threads gracefully...")
        _shutdown_event.set()
        _stop_dashboard.set()

    signal.signal(signal.SIGINT,  _main_shutdown)
    signal.signal(signal.SIGTERM, _main_shutdown)

    # One-time MT5 initialisation (connection is process-wide)
    _init_logger = Logger()
    _init_config = Config()
    _init_fetcher = DataFetcher(_init_config, _init_logger)
    if not _init_fetcher.initialize_mt5():
        _init_logger.error("Failed to initialize MT5 -- exiting.")
        sys.exit(1)

    # Print magic number table
    _init_logger.info("=" * 55)
    _init_logger.info("  MAGIC NUMBER REFERENCE TABLE (restart-safe)")
    _init_logger.info("=" * 55)
    _init_config.config["default"]["num_symbols"] = len(args.symbols)
    for sym in args.symbols:
        configured = _init_config.get('magic_number', None, symbol=sym)
        if configured:
            magic = int(configured); source = "config"
        else:
            _b = hashlib.md5(sym.encode()).digest()
            magic = 100000 + (int.from_bytes(_b[:4], 'big') % 900000)
            source = "auto"
        _init_logger.info(f"  {sym:<14} -> {magic}  ({source})")
    _init_logger.info("=" * 55)

    # Initialise dashboard state for all symbols
    for sym in args.symbols:
        update_dashboard(sym, status='waiting')

    # Start dashboard thread (_stop_dashboard already created above near signal handlers)
    if not args.no_dashboard:
        _dash_thread = threading.Thread(
            target=run_dashboard,
            args=(args.symbols, _stop_dashboard, args.dashboard_refresh, args.dashboard_port),
            daemon=True,
            name="dashboard",
        )
        _dash_thread.start()

    # Launch one thread per symbol -- each gets its own event loop so
    # CPU-bound training in one symbol never blocks the others.
    _init_logger.info(
        f"Launching {len(args.symbols)} symbol threads: {', '.join(args.symbols)}"
    )

    # Write initial symbols.json so user can edit it to add/remove symbols at runtime
    _symbols_file = "symbols.json"
    def _write_symbols_file(syms):
        try:
            with open(_symbols_file, 'w') as _f:
                json.dump({
                    "symbols": syms,
                    "_comment": "Edit this file to add or remove symbols while the bot is running. "
                                "Changes are detected within symbol_watch_interval seconds."
                }, _f, indent=2)
        except Exception as _e:
            _init_logger.warning(f"Could not write {_symbols_file}: {_e}")

    _write_symbols_file(args.symbols)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=64,   # generous cap; each symbol uses one thread
        thread_name_prefix="bot",
    )
    _active_futures: dict = {}   # symbol -> Future
    _active_symbols: set  = set()

    def _launch_symbol(sym):
        """Start a new thread for sym if not already running."""
        if sym in _active_symbols:
            return
        _active_symbols.add(sym)
        update_dashboard(sym, status='starting')
        _active_futures[sym] = executor.submit(run_symbol_in_thread, sym, args.backtest)
        _init_logger.info(f"[SymbolManager] Started thread for {sym}")

    def _remove_symbol(sym):
        """Signal a running symbol thread to stop gracefully."""
        if sym not in _active_symbols:
            return
        if sym in _symbol_stop_events:
            _symbol_stop_events[sym].set()
            update_dashboard(sym, status='removing...')
            _init_logger.info(f"[SymbolManager] Signalled {sym} to stop")

    # Launch initial symbols
    for sym in args.symbols:
        _launch_symbol(sym)

    # Symbol watcher: polls symbols.json every symbol_watch_interval seconds
    _watch_interval = _init_config.get('symbol_watch_interval', 10)

    def _symbol_watcher():
        _last_mtime = None
        while not _shutdown_event.is_set():
            try:
                if os.path.exists(_symbols_file):
                    mtime = os.path.getmtime(_symbols_file)
                    if mtime != _last_mtime:
                        _last_mtime = mtime
                        with open(_symbols_file) as _f:
                            data = json.load(_f)
                        desired = set(data.get("symbols", []))
                        current = set(_active_symbols)

                        added   = desired - current
                        removed = current - desired

                        for sym in added:
                            _init_logger.info(f"[SymbolManager] Adding new symbol: {sym}")
                            _launch_symbol(sym)

                        for sym in removed:
                            _init_logger.info(f"[SymbolManager] Removing symbol: {sym}")
                            _remove_symbol(sym)
            except Exception as _e:
                _init_logger.warning(f"[SymbolManager] Watch error: {_e}")
            time.sleep(_watch_interval)

    _watcher_thread = threading.Thread(
        target=_symbol_watcher, daemon=True, name="symbol-watcher"
    )
    _watcher_thread.start()
    _init_logger.info(
        f"[SymbolManager] Watching {_symbols_file} for changes every {_watch_interval}s. "
        f"Edit that file to add/remove symbols at runtime."
    )

    # Poll futures in the main thread -- stays responsive to Ctrl+C and hot symbol changes
    try:
        while not _shutdown_event.is_set():
            # Reap completed futures
            done_syms = [sym for sym, f in list(_active_futures.items()) if f.done()]
            for sym in done_syms:
                fut = _active_futures.pop(sym)
                _active_symbols.discard(sym)
                try:
                    fut.result()
                    _init_logger.info(f"{sym} thread finished normally.")
                except Exception as exc:
                    _init_logger.error(f"{sym} thread raised: {exc}", exc_info=True)
                    update_dashboard(sym, status='ERROR', error=str(exc)[:60])

            if not _active_futures and not _shutdown_event.is_set():
                _init_logger.info("All symbol threads finished -- exiting.")
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        _init_logger.info("KeyboardInterrupt -- shutting down...")
        _shutdown_event.set()
        _stop_dashboard.set()
        executor.shutdown(wait=False, cancel_futures=True)

    if _shutdown_event.is_set() and _active_futures:
        _init_logger.info(f"Waiting up to 30s for {len(_active_futures)} thread(s) to wind down...")
        executor.shutdown(wait=True)

    _stop_dashboard.set()
    _init_logger.info("All symbol threads finished.")