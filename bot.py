"""
MEC Paper Trading Bot
=====================
- Descarga vela diaria en curso (live) cada ejecucion
- Calcula senal MEC (DC + Head & Shoulders + MA filtro)
- Gestiona dos cuentas en papel:
    Cuenta A (40 EUR): estrategia M1 - leverage 3x + vol sizing
    Cuenta B (160 EUR): estrategia S4 - filtro momentum, leverage dinamico 2x-5x
- Guarda estado en state.json y publica snapshot en docs/data.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────
# CONFIGURACION DEL PORTFOLIO
# ─────────────────────────────────────────────
CAPITAL_A = 40.0
CAPITAL_B = 160.0

STATE_FILE = Path("state.json")
DATA_FILE  = Path("docs/data.json")

SIGMA          = 0.008
ORDER          = 3
SHOULDER_TOL   = 0.25
NECK_TOL       = 0.02
LOOKAHEAD      = 30
FAST_MA        = 20
SLOW_MA        = 50
WINDOW_CONFIRM = 5
ATR_PERIOD     = 14
SL_ATR         = 2.0
TP_ATR         = 1.5
HORIZON_DAYS   = 30
FEE_BPS        = 5.0


# ─────────────────────────────────────────────
# HELPERS DE COLUMNAS (fuera de cualquier funcion)
# ─────────────────────────────────────────────
def _get_series(df, names):
    """Devuelve una Series de df buscando por nombre, soporta MultiIndex."""
    for n in names:
        if n in df.columns:
            s = df[n]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.squeeze()
    for n in names:
        cols = [c for c in df.columns
                if (isinstance(c, tuple) and c[0] == n) or c == n]
        if cols:
            s = df[cols[0]]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.squeeze()
    return pd.Series(dtype=float)


def _get_float(df, names, idx=0):
    """Devuelve un float de la fila idx de la columna encontrada."""
    s = _get_series(df, names)
    if s.empty:
        return np.nan
    if idx == -1:
        return float(s.iloc[-1])
    return float(s.iloc[idx])


# ─────────────────────────────────────────────
# DESCARGA: vela diaria historica + vela viva
# ─────────────────────────────────────────────
def _clean_df(df):
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cands = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        scores = [sum(str(x) in cands for x in out.columns.get_level_values(l))
                  for l in range(out.columns.nlevels)]
        out.columns = out.columns.get_level_values(int(np.argmax(scores)))
    for old, new in [("Adj Close", "Close"), ("AdjClose", "Close")]:
        if old in out.columns and "Close" not in out.columns:
            out = out.rename(columns={old: new})
        elif old in out.columns:
            out = out.drop(columns=[old])
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in out.columns:
            raise ValueError(f"Falta columna: {c}")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=needed)


def download_with_live_candle(ticker="BTC-USD", start="2010-01-01"):
    print("  Descargando historico diario...")
    daily = yf.download(ticker, start=start, interval="1d",
                        auto_adjust=False, progress=False)

    print("  Descargando datos intradia (5m, ultimos 7 dias)...")
    intra = yf.download(ticker, period="7d", interval="5m",
                        auto_adjust=False, progress=False)

    if intra is None or intra.empty:
        print("  AVISO: sin datos intradia, usando solo historico")
        return _clean_df(daily)

    idx = intra.index
    try:
        idx = idx.tz_convert(None)
    except Exception:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
    intra = intra.copy()
    intra.index = idx

    last_day  = intra.index[-1].date()
    day_slice = intra[intra.index.date == last_day]
    if day_slice.empty:
        return _clean_df(daily)

    live_open  = float(_get_series(day_slice, ["Open"]).iloc[0])
    live_high  = float(_get_series(day_slice, ["High"]).max())
    live_low   = float(_get_series(day_slice, ["Low"]).min())
    live_close = float(_get_series(day_slice, ["Close"]).iloc[-1])
    live_vol   = float(_get_series(day_slice, ["Volume"]).sum())

    live_row = pd.DataFrame({
        "Open":      [live_open],
        "High":      [live_high],
        "Low":       [live_low],
        "Close":     [live_close],
        "Adj Close": [live_close],
        "Volume":    [live_vol],
    }, index=[pd.Timestamp(last_day)])

    out = _clean_df(daily)
    if pd.Timestamp(last_day) in out.index:
        for col in live_row.columns:
            if col in out.columns:
                out.loc[pd.Timestamp(last_day), col] = live_row.iloc[0][col]
    else:
        out = pd.concat([out, live_row]).sort_index()

    print(f"  Vela live: {last_day} O={live_open:.0f} H={live_high:.0f} "
          f"L={live_low:.0f} C={live_close:.0f}")
    return out


# ─────────────────────────────────────────────
# PIPELINE MEC
# ─────────────────────────────────────────────
def directional_change(close, high, low, sigma):
    up_zig = True
    tmp_max = high[0]; tmp_min = low[0]
    tmp_max_i = 0;     tmp_min_i = 0
    tops = []; bottoms = []
    for i in range(len(close)):
        if up_zig:
            if high[i] > tmp_max:
                tmp_max = high[i]; tmp_max_i = i
            elif close[i] < tmp_max * (1 - sigma):
                tops.append([i, tmp_max_i, tmp_max])
                up_zig = False
                tmp_min = low[i]; tmp_min_i = i
        else:
            if low[i] < tmp_min:
                tmp_min = low[i]; tmp_min_i = i
            elif close[i] > tmp_min * (1 + sigma):
                bottoms.append([i, tmp_min_i, tmp_min])
                up_zig = True
                tmp_max = high[i]; tmp_max_i = i
    return tops, bottoms


def dc_labels(df, sigma=SIGMA):
    c = df["Close"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    tops, bottoms = directional_change(c, h, l, sigma)
    lab = np.zeros(len(df), dtype=np.int8)
    for (i, _, _) in tops:
        lab[min(i, len(lab) - 1)] = 2
    for (i, _, _) in bottoms:
        lab[min(i, len(lab) - 1)] = 1
    return pd.Series(lab, index=df.index, name="label_dc")


def hs_labels(df, order=ORDER, shoulder_tol=SHOULDER_TOL,
              neck_tol=NECK_TOL, lookahead=LOOKAHEAD):
    c = df["Close"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    n = len(c)
    lab = np.zeros(n, dtype=np.int8)

    def lmax():
        r = []
        for i in range(order, n - order):
            if h[i] == max(h[i - order:i + order + 1]):
                r.append(i)
        return r

    def lmin():
        r = []
        for i in range(order, n - order):
            if l[i] == min(l[i - order:i + order + 1]):
                r.append(i)
        return r

    maxima = lmax()
    minima = lmin()

    for k in range(1, len(maxima) - 1):
        ls, hd, rs = maxima[k - 1], maxima[k], maxima[k + 1]
        if hd <= ls or hd >= rs:
            continue
        if h[hd] <= h[ls] or h[hd] <= h[rs]:
            continue
        if abs(h[ls] - h[rs]) / h[hd] > shoulder_tol:
            continue
        mb  = [m for m in minima if ls < m < hd]
        mb2 = [m for m in minima if hd < m < rs]
        if not mb or not mb2:
            continue
        neck = (l[mb[-1]] + l[mb2[0]]) / 2
        for j in range(rs + 1, min(rs + lookahead + 1, n)):
            if c[j] < neck * (1 - neck_tol):
                lab[j] = 2
                break

    for k in range(1, len(minima) - 1):
        ls, hd, rs = minima[k - 1], minima[k], minima[k + 1]
        if hd <= ls or hd >= rs:
            continue
        if l[hd] >= l[ls] or l[hd] >= l[rs]:
            continue
        if abs(l[ls] - l[rs]) / abs(l[hd]) > shoulder_tol:
            continue
        mb  = [m for m in maxima if ls < m < hd]
        mb2 = [m for m in maxima if hd < m < rs]
        if not mb or not mb2:
            continue
        neck = (h[mb[-1]] + h[mb2[0]]) / 2
        for j in range(rs + 1, min(rs + lookahead + 1, n)):
            if c[j] > neck * (1 + neck_tol):
                lab[j] = 1
                break

    return pd.Series(lab, index=df.index, name="label_hs")


def add_signals(df, fast=FAST_MA, slow=SLOW_MA, window=WINDOW_CONFIRM):
    df = df.copy()
    cl = pd.to_numeric(df["Close"], errors="coerce")
    df["MA_fast"] = cl.rolling(fast, min_periods=1).mean()
    df["MA_slow"] = cl.rolling(slow, min_periods=1).mean()
    df["label_dc"] = dc_labels(df)
    df["label_hs"] = hs_labels(df)

    n = len(df)
    sig = np.zeros(n, dtype=np.int8)
    for i in range(n):
        j0 = max(0, i - window + 1)
        dc = df["label_dc"].iloc[j0:i + 1]
        hs = df["label_hs"].iloc[j0:i + 1]
        long_ok  = dc.eq(1).any() or hs.eq(1).any()
        short_ok = dc.eq(2).any() or hs.eq(2).any()
        tu = df["MA_fast"].iloc[i] > df["MA_slow"].iloc[i]
        td = df["MA_fast"].iloc[i] < df["MA_slow"].iloc[i]
        if long_ok and tu and not (short_ok and td):
            sig[i] = 1
        elif short_ok and td and not (long_ok and tu):
            sig[i] = 2

    df["signal_final"] = sig.astype(np.int8)

    ev = np.zeros_like(sig, dtype=np.int8)
    prev = 0
    for i, s in enumerate(sig):
        if s in (1, 2) and s != prev:
            ev[i] = s
        prev = s
    df["signal_event"] = ev
    return df


def add_indicators(df):
    df = df.copy()
    cl = df["Close"].astype(float)

    delta = cl.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    df["ROC10"] = cl.pct_change(10) * 100

    vol = df["Volume"].astype(float)
    df["VOL_REL"] = vol / vol.rolling(20).mean()

    df["MA200"] = cl.rolling(200, min_periods=50).mean()

    hi = df["High"].astype(float)
    lo = df["Low"].astype(float)
    pc = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - pc).abs(), (lo - pc).abs()],
                   axis=1).max(axis=1)
    df["ATR14"]   = tr.rolling(ATR_PERIOD).mean()
    df["ATR_PCT"] = df["ATR14"] / cl

    ss = np.zeros(len(df), dtype=np.int8)
    for i in range(len(df)):
        j0 = max(0, i - 4)
        dc_hit = df["label_dc"].iloc[j0:i + 1].isin([1, 2]).any()
        hs_hit = df["label_hs"].iloc[j0:i + 1].isin([1, 2]).any()
        ss[i] = int(dc_hit) + int(hs_hit) * 2
    df["SIG_STRENGTH"] = ss

    rsi_n  = (df["RSI"] - 50) / 50
    roc_n  = df["ROC10"].clip(-30, 30) / 30
    reg_sc = np.sign(cl - df["MA200"]).fillna(0)
    df["MOMENTUM_SCORE"] = rsi_n * 0.4 + roc_n * 0.4 + reg_sc * 0.2

    return df


# ─────────────────────────────────────────────
# SIZING
# ─────────────────────────────────────────────
def sizing_m1(row, side):
    atr_pct = float(row.get("ATR_PCT", 0.02))
    if pd.isna(atr_pct) or atr_pct <= 0:
        atr_pct = 0.02
    frac = float(np.clip(0.20 * (0.015 / atr_pct), 0.05, 0.30))
    return 3.0, frac


def sizing_s4(row, side):
    mom     = float(row.get("MOMENTUM_SCORE", 0))
    atr_pct = float(row.get("ATR_PCT", 0.025))
    if pd.isna(mom):
        mom = 0
    if pd.isna(atr_pct) or atr_pct <= 0:
        atr_pct = 0.025
    alignment = mom if side == 1 else -mom
    if alignment < -0.10:
        return 2.0, 0.05
    leverage = float(np.clip(2.0 + max(alignment, 0) * 3.0, 2.0, 5.0))
    frac     = float(np.clip(0.20 * (0.015 / atr_pct), 0.05, 0.30))
    return leverage, frac


# ─────────────────────────────────────────────
# ESTADO
# ─────────────────────────────────────────────
def make_account(name, capital):
    return {
        "name":           name,
        "initial":        capital,
        "cash":           capital,
        "equity":         capital,
        "position":       None,
        "trades":         [],
        "equity_history": [],
    }


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            st = json.load(f)
        for acc in [st["account_a"], st["account_b"]]:
            acc.setdefault("equity_history", [])
        return st
    return {
        "account_a":   make_account("M1 (40EUR)",  CAPITAL_A),
        "account_b":   make_account("S4 (160EUR)", CAPITAL_B),
        "last_signal": 0,
        "last_run":    None,
        "runs":        0,
    }


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2, default=str)


# ─────────────────────────────────────────────
# LOGICA DE TRADE
# ─────────────────────────────────────────────
def open_trade(account, side, entry_px, atr, sizing_fn, row):
    if account["position"] is not None:
        return

    leverage, frac = sizing_fn(row, side)
    leverage = float(np.clip(leverage, 1.0, 10.0))
    frac     = float(np.clip(frac, 0.02, 0.40))

    margin   = account["cash"] * frac
    notional = margin * leverage
    units    = notional / entry_px
    fee      = notional * (FEE_BPS / 10_000)

    if margin + fee > account["cash"]:
        print(f"    [{account['name']}] Sin capital suficiente")
        return

    sl_dist = SL_ATR * atr
    tp_dist = TP_ATR * atr

    if side == 1:
        sl_px = entry_px - sl_dist
        tp_px = entry_px + tp_dist
    else:
        sl_px = entry_px + sl_dist
        tp_px = entry_px - tp_dist

    account["cash"] -= (margin + fee)
    account["position"] = {
        "side":      "LONG" if side == 1 else "SHORT",
        "side_int":  side,
        "entry_px":  entry_px,
        "units":     units,
        "margin":    margin,
        "notional":  notional,
        "leverage":  leverage,
        "sl_px":     sl_px,
        "tp_px":     tp_px,
        "entry_fee": fee,
        "open_date": datetime.now(timezone.utc).isoformat(),
        "open_bars": 0,
    }
    print(f"    [{account['name']}] ABRE {account['position']['side']} "
          f"@ {entry_px:.0f}  SL={sl_px:.0f}  TP={tp_px:.0f}  "
          f"lev={leverage:.1f}x  margin={margin:.2f}")


def check_close_trade(account, row):
    pos = account["position"]
    if pos is None:
        return

    high_px  = float(row["High"])
    low_px   = float(row["Low"])
    close_px = float(row["Close"])
    side     = pos["side_int"]

    pos["open_bars"] += 1

    sl_hit = (side == 1 and low_px  <= pos["sl_px"]) or \
             (side == 2 and high_px >= pos["sl_px"])
    tp_hit = (side == 1 and high_px >= pos["tp_px"]) or \
             (side == 2 and low_px  <= pos["tp_px"])
    time_exit = pos["open_bars"] >= HORIZON_DAYS

    reason  = None
    exit_px = None

    if sl_hit and tp_hit:
        reason  = "SL"
        exit_px = pos["sl_px"]
    elif sl_hit:
        reason  = "SL"
        exit_px = pos["sl_px"]
    elif tp_hit:
        reason  = "TP"
        exit_px = pos["tp_px"]
    elif time_exit:
        reason  = "TIME_EXIT"
        exit_px = close_px

    if reason:
        _close_trade(account, exit_px, reason)


def _close_trade(account, exit_px, reason):
    pos  = account["position"]
    side = pos["side_int"]

    if side == 1:
        pnl_raw = (exit_px - pos["entry_px"]) * pos["units"]
    else:
        pnl_raw = (pos["entry_px"] - exit_px) * pos["units"]

    exit_fee = exit_px * pos["units"] * (FEE_BPS / 10_000)
    pnl_net  = pnl_raw - pos["entry_fee"] - exit_fee

    account["cash"] += pos["margin"] + pnl_net
    account["cash"]  = max(account["cash"], 0.0)

    trade_record = {
        "side":        pos["side"],
        "entry_px":    round(pos["entry_px"], 2),
        "exit_px":     round(exit_px, 2),
        "exit_reason": reason,
        "pnl_net":     round(pnl_net, 4),
        "pnl_pct":     round(pnl_net / pos["margin"] * 100, 2),
        "leverage":    round(pos["leverage"], 2),
        "open_date":   pos["open_date"],
        "close_date":  datetime.now(timezone.utc).isoformat(),
    }
    account["trades"].append(trade_record)
    account["position"] = None

    sign = "+" if pnl_net >= 0 else ""
    print(f"    [{account['name']}] CIERRA {pos['side']} @ {exit_px:.0f} "
          f"-> {reason}  PnL={sign}{pnl_net:.4f} ({sign}{trade_record['pnl_pct']:.1f}%)")


def update_equity(account, current_price):
    pos = account["position"]
    if pos:
        side   = pos["side_int"]
        unreal = (current_price - pos["entry_px"]) * pos["units"] if side == 1 \
                 else (pos["entry_px"] - current_price) * pos["units"]
        account["equity"] = max(account["cash"] + pos["margin"] + unreal, 0.0)
    else:
        account["equity"] = account["cash"]

    account["equity_history"].append({
        "ts":     datetime.now(timezone.utc).isoformat(),
        "equity": round(account["equity"], 4),
    })
    if len(account["equity_history"]) > 1000:
        account["equity_history"] = account["equity_history"][-1000:]


# ─────────────────────────────────────────────
# PUBLICAR SNAPSHOT
# ─────────────────────────────────────────────
def publish_data(state, df, current_signal):
    def _acc_snapshot(acc):
        pos    = acc["position"]
        closed = list(acc["trades"])
        wins   = [t for t in closed if t["pnl_net"] > 0]
        total_pnl = sum(t["pnl_net"] for t in closed)
        wr = len(wins) / len(closed) * 100 if closed else 0.0

        pos_snap = None
        if pos:
            last_close = float(df["Close"].iloc[-1])
            side = pos["side_int"]
            unreal = (last_close - pos["entry_px"]) * pos["units"] if side == 1 \
                     else (pos["entry_px"] - last_close) * pos["units"]
            pos_snap = {
                "side":           pos["side"],
                "entry_px":       round(pos["entry_px"], 0),
                "sl_px":          round(pos["sl_px"], 0),
                "tp_px":          round(pos["tp_px"], 0),
                "leverage":       round(pos["leverage"], 1),
                "pnl_unreal":     round(unreal, 4),
                "pnl_unreal_pct": round(unreal / pos["margin"] * 100, 2),
                "open_date":      pos["open_date"],
            }

        return {
            "name":           acc["name"],
            "initial":        round(acc["initial"], 2),
            "equity":         round(acc["equity"], 4),
            "cash":           round(acc["cash"], 4),
            "total_pnl":      round(total_pnl, 4),
            "total_pnl_pct":  round((acc["equity"] / acc["initial"] - 1) * 100, 2),
            "num_trades":     len(closed),
            "win_rate":       round(wr, 1),
            "position":       pos_snap,
            "trades":         closed[-50:],
            "equity_history": acc["equity_history"][-200:],
        }

    last      = df.iloc[-1]
    btc_price = round(float(last["Close"]), 2)
    sig_map   = {0: "neutral", 1: "LONG", 2: "SHORT"}
    sig_text  = sig_map.get(current_signal, "neutral")

    recent_signals = df[df["signal_event"].isin([1, 2])].tail(20)
    signal_history = [
        {
            "date":   str(d.date()),
            "signal": "LONG" if int(v) == 1 else "SHORT",
            "price":  round(float(df.loc[d, "Close"]), 0),
        }
        for d, v in recent_signals["signal_event"].items()
    ]

    price_history = [
        {"date": str(d.date()), "close": round(float(df.loc[d, "Close"]), 0)}
        for d in df.index[-90:]
        if not pd.isna(df.loc[d, "Close"])
    ]

    data = {
        "updated_at":       datetime.now(timezone.utc).isoformat(),
        "btc_price":        btc_price,
        "btc_change_24h":   round(
            (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100, 2
        ) if len(df) >= 2 else 0.0,
        "current_signal":   sig_text,
        "last_signal_date": str(df[df["signal_event"].isin([1, 2])].index[-1].date())
                            if df["signal_event"].isin([1, 2]).any() else None,
        "runs":             state["runs"],
        "account_a":        _acc_snapshot(state["account_a"]),
        "account_b":        _acc_snapshot(state["account_b"]),
        "signal_history":   signal_history,
        "price_history":    price_history,
        "portfolio_equity": round(
            state["account_a"]["equity"] + state["account_b"]["equity"], 4
        ),
        "portfolio_initial": CAPITAL_A + CAPITAL_B,
    }

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=str)

    print(f"  docs/data.json actualizado — BTC={btc_price}  senal={sig_text}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    print(f"\n{'='*50}")
    print(f"  MEC Bot — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*50}")

    state = load_state()
    state["runs"] += 1
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    print(f"  Ejecucion #{state['runs']}")

    try:
        df = download_with_live_candle()
    except Exception as e:
        print(f"  ERROR descargando datos: {e}")
        return

    print("  Calculando senal MEC...")
    df = add_signals(df)
    df = add_indicators(df)

    last_row    = df.iloc[-1]
    current_px  = float(last_row["Close"])
    current_sig = int(last_row["signal_event"])
    atr_val     = float(last_row["ATR14"]) if not pd.isna(last_row["ATR14"]) \
                  else current_px * 0.02
    prev_signal = state.get("last_signal", 0)

    sig_str = "LONG" if current_sig == 1 else "SHORT" if current_sig == 2 else "neutral"
    print(f"  Precio actual: {current_px:.0f}  |  Senal hoy: {sig_str}  |  ATR: {atr_val:.0f}")

    for acc in [state["account_a"], state["account_b"]]:
        check_close_trade(acc, last_row)

    if current_sig in (1, 2) and current_sig != prev_signal:
        print(f"  *** SENAL NUEVA: {sig_str} ***")
        open_trade(state["account_a"], current_sig, current_px, atr_val, sizing_m1, last_row)
        open_trade(state["account_b"], current_sig, current_px, atr_val, sizing_s4, last_row)
        state["last_signal"] = current_sig
    elif current_sig == 0 and prev_signal != 0:
        state["last_signal"] = 0

    for acc in [state["account_a"], state["account_b"]]:
        update_equity(acc, current_px)

    pnl_total = (state["account_a"]["equity"] + state["account_b"]["equity"]
                 - CAPITAL_A - CAPITAL_B)
    sign = "+" if pnl_total >= 0 else ""
    print(f"  Portfolio: {state['account_a']['equity']:.2f} + "
          f"{state['account_b']['equity']:.2f}  "
          f"PnL total: {sign}{pnl_total:.4f}")

    save_state(state)
    publish_data(state, df, current_sig)
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run()
