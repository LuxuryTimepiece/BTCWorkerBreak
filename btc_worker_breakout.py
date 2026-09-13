#!/usr/bin/env python3
"""
BTC Breakout Worker
===================
Philosophy: Trade volatility expansions / range breakouts.
- Entry when price breaks above recent 4H range high with expanding ATR and rising volume proxy
- MACD hist positive and expanding as confirmation
- Daily regime still used as a soft filter (prefer FRIENDLY/MIXED)
- Risk 1.25%, ATR stop 1.6x

Independent worker. Own state, own equity, own /status.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from flask import Flask, jsonify
import json, time, threading, os, warnings
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore")

SYMBOL = "BTC-USD"
INTERVAL = "4h"
RANGE_LOOKBACK = 12          # ~2 days of 4H bars
EMA_FAST = 12
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_LEN = 14
ATR_LEN, ATR_STOP_MULT = 14, 1.6
TRAIL_ACTIVATE_R, TRAIL_ATR_MULT = 0.7, 1.5
RISK_PCT = 0.0125
FEE_RT = 0.0004
INITIAL_EQUITY = 10_000.0
CHECK_INTERVAL = 300
DAILY_EMA_LEN = 50
MAX_DRAWDOWN_PCT = 0.08
MAX_CONSECUTIVE_LOSSES = 3
MIN_ATR_PCT = 0.0045
MAX_BARS = 48

STATE_FILE = Path("paper_state_breakout.json")

def default_state():
    return {
        "worker": "breakout",
        "philosophy": "volatility expansion / range breakout",
        "equity": INITIAL_EQUITY,
        "peak_equity": INITIAL_EQUITY,
        "position": 0,
        "entry_price": 0.0,
        "stop_price": 0.0,
        "orig_stop_pct": 0.0,
        "bars_held": 0,
        "last_bar_time": None,
        "trades": [],
        "equity_curve": [{"time": datetime.now(timezone.utc).isoformat(), "equity": INITIAL_EQUITY}],
        "last_update": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "regime": "unknown",
        "risk_state": "unknown",
        "agent_decision": "unknown",
        "last_price": None,
    }

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            for k, v in default_state().items():
                if k not in state:
                    state[k] = v
            return state
        except Exception:
            pass
    return default_state()

def save_state(state):
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def _norm(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df

def fetch_4h():
    df = yf.download(SYMBOL, period="60d", interval=INTERVAL, progress=False)
    if df.empty:
        return None
    df = _norm(df.dropna())
    df["ema12"] = EMAIndicator(df["close"], EMA_FAST).ema_indicator()
    macd = MACD(df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["rsi"] = RSIIndicator(df["close"], RSI_LEN).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], ATR_LEN).average_true_range()
    df["atr_prev"] = df["atr"].shift(1)
    # Rolling range high (exclude current bar for clean breakout)
    df["range_high"] = df["high"].shift(1).rolling(RANGE_LOOKBACK).max()
    return df.dropna()

def fetch_daily():
    df = yf.download(SYMBOL, period="1y", interval="1d", progress=False)
    if df.empty:
        return None
    df = _norm(df.dropna())
    df["ema50"] = EMAIndicator(df["close"], DAILY_EMA_LEN).ema_indicator()
    return df.dropna()

def evaluate_regime():
    df = fetch_daily()
    if df is None or len(df) < DAILY_EMA_LEN + 5:
        return "unknown", {}
    last, prev = df.iloc[-1], df.iloc[-6] if len(df) >= 6 else df.iloc[0]
    price, ema50 = float(last["close"]), float(last["ema50"])
    slope_up = ema50 >= float(prev["ema50"])
    above = price > ema50
    if above and slope_up:
        regime = "FRIENDLY"
    elif (not above) or (not slope_up and price < ema50 * 0.98):
        regime = "HOSTILE"
    else:
        regime = "MIXED"
    return regime, {"daily_close": round(price, 2), "daily_ema50": round(ema50, 2),
                    "ema50_slope_up": slope_up, "price_above_ema50": above}

def evaluate_risk(state):
    equity = float(state.get("equity", INITIAL_EQUITY))
    peak = float(state.get("peak_equity", equity))
    if equity > peak:
        peak = equity
        state["peak_equity"] = peak
    drawdown = (peak - equity) / peak if peak > 0 else 0.0
    consecutive = 0
    for t in reversed(state.get("trades", [])):
        if t.get("pnl_pct", 0) < 0:
            consecutive += 1
        else:
            break
    risk_off = (drawdown >= MAX_DRAWDOWN_PCT) or (consecutive >= MAX_CONSECUTIVE_LOSSES)
    return ("RISK-OFF" if risk_off else "RISK-ON",
            {"equity": round(equity, 2), "peak_equity": round(peak, 2),
             "drawdown_pct": round(drawdown * 100, 2), "consecutive_losses": consecutive})

def get_signal(row):
    """Breakout: close above recent range high + ATR expanding + MACD hist positive & expanding."""
    close = float(row["close"])
    range_high = float(row["range_high"])
    atr = float(row["atr"])
    atr_prev = float(row["atr_prev"])
    hist = float(row["macd_hist"])
    hist_prev = float(row["macd_hist_prev"])
    breakout = close > range_high
    atr_expanding = atr > atr_prev
    macd_ok = (hist > 0) and (hist > hist_prev)
    return bool(breakout and atr_expanding and macd_ok)

def run_once(state):
    df = fetch_4h()
    if df is None or len(df) < RANGE_LOOKBACK + 5:
        state["status"] = "data_error"
        return state
    latest = df.iloc[-1]
    bar_time = str(latest.name)
    price = float(latest["close"])
    atr = float(latest["atr"])
    atr_pct = atr / price if price > 0 else 0

    regime, regime_detail = evaluate_regime()
    risk_state, risk_detail = evaluate_risk(state)
    state["regime"] = regime
    state["risk_state"] = risk_state
    state["regime_detail"] = regime_detail
    state["risk_detail"] = risk_detail
    allow = (regime in ("FRIENDLY", "MIXED")) and (risk_state == "RISK-ON")
    state["agent_decision"] = "CONTINUE" if allow else "PAUSE"

    if state.get("last_bar_time") == bar_time:
        state["status"] = "waiting_for_new_bar"
        state["last_price"] = price
        return state

    if state["position"] == 1:
        state["bars_held"] = state.get("bars_held", 0) + 1
        entry, stop = state["entry_price"], state["stop_price"]
        r_mult = (price - entry) / (entry - stop) if entry > stop else 0
        if r_mult >= TRAIL_ACTIVATE_R:
            new_stop = price - atr * TRAIL_ATR_MULT
            if new_stop > stop:
                state["stop_price"] = new_stop
        exit_now, reason = False, ""
        if price <= state["stop_price"]:
            exit_now, reason = True, "stop"
        elif float(latest["macd_hist"]) < 0 and float(latest["macd_hist"]) < float(latest["macd_hist_prev"]):
            exit_now, reason = True, "momentum_fail"
        elif state["bars_held"] >= MAX_BARS:
            exit_now, reason = True, "time"
        if exit_now:
            raw = (price - entry) / entry
            pnl = raw - FEE_RT
            risk_d = state["equity"] * RISK_PCT
            size = risk_d / state["orig_stop_pct"] if state.get("orig_stop_pct", 0) > 1e-8 else 0
            dollar = size * pnl
            state["equity"] += dollar
            if state["equity"] > state.get("peak_equity", state["equity"]):
                state["peak_equity"] = state["equity"]
            state.setdefault("trades", []).append({
                "time": bar_time, "side": "long",
                "entry": round(entry, 2), "exit": round(price, 2),
                "pnl_pct": round(pnl * 100, 3), "dollar": round(dollar, 2),
                "reason": reason, "bars": state["bars_held"],
                "equity_after": round(state["equity"], 2),
            })
            state["position"] = 0
            state["bars_held"] = 0

    weekend = latest.name.weekday() >= 5 if hasattr(latest.name, "weekday") else False
    atr_ok = atr_pct >= MIN_ATR_PCT
    if state["position"] == 0 and allow and atr_ok and not weekend:
        if get_signal(latest) and atr > 0:
            stop_dist = max(atr * ATR_STOP_MULT, price * MIN_ATR_PCT * 1.5)
            state["position"] = 1
            state["entry_price"] = price
            state["stop_price"] = price - stop_dist
            state["orig_stop_pct"] = stop_dist / price
            state["bars_held"] = 0

    state["last_bar_time"] = bar_time
    state.setdefault("equity_curve", []).append({
        "time": datetime.now(timezone.utc).isoformat(), "equity": state["equity"]
    })
    if len(state["equity_curve"]) > 3000:
        state["equity_curve"] = state["equity_curve"][-2000:]
    state["status"] = "ok"
    state["last_price"] = price
    return state

def worker_loop():
    print("Breakout worker loop started", flush=True)
    while True:
        try:
            state = load_state()
            state = run_once(state)
            save_state(state)
            pos = "LONG" if state["position"] == 1 else "FLAT"
            print(f"[breakout {datetime.now().strftime('%H:%M:%S')}] {pos} | Eq ${state['equity']:.2f} | "
                  f"{state.get('regime')} / {state.get('risk_state')} / {state.get('agent_decision')}", flush=True)
        except Exception as e:
            print(f"Breakout worker error: {e}", flush=True)
        time.sleep(CHECK_INTERVAL)

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"service": "BTC Breakout Worker", "endpoints": {"/status": "state", "/health": "health"}})

@app.route("/health")
def health():
    return jsonify({"status": "alive", "worker": "breakout", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/status")
def status():
    return jsonify(load_state())

_started = False
_lock = threading.Lock()
def start_once():
    global _started
    with _lock:
        if not _started:
            threading.Thread(target=worker_loop, daemon=True).start()
            _started = True
start_once()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8083))
    app.run(host="0.0.0.0", port=port)
