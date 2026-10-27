import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ============================================================
# GPR CRYPTO FUTURES SCANNER
# Version: 15m
# Public OKX market data - NO API KEY
# ============================================================

st.set_page_config(
    page_title="GPR Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://openapi.okx.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

TIMEOUT = 12

# Fallback universe.
# Primary universe tetap diambil dari OKX ticker SWAP.
FALLBACK_SYMBOLS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "ADA-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "LINK-USDT-SWAP",
    "DOT-USDT-SWAP",
    "SUI-USDT-SWAP",
    "LTC-USDT-SWAP",
    "BCH-USDT-SWAP",
    "UNI-USDT-SWAP",
    "AAVE-USDT-SWAP",
    "ETC-USDT-SWAP",
    "ATOM-USDT-SWAP",
    "NEAR-USDT-SWAP",
    "APT-USDT-SWAP",
    "ARB-USDT-SWAP",
    "OP-USDT-SWAP",
    "FIL-USDT-SWAP",
    "INJ-USDT-SWAP",
    "TIA-USDT-SWAP",
    "SEI-USDT-SWAP",
    "TON-USDT-SWAP",
    "TRX-USDT-SWAP",
    "WLD-USDT-SWAP",
    "PEPE-USDT-SWAP",
    "SHIB-USDT-SWAP",
    "MKR-USDT-SWAP",
    "RUNE-USDT-SWAP",
    "KAS-USDT-SWAP",
    "JUP-USDT-SWAP",
    "WIF-USDT-SWAP",
    "BONK-USDT-SWAP",
    "FLOKI-USDT-SWAP",
    "ENA-USDT-SWAP",
    "ONDO-USDT-SWAP",
    "PENDLE-USDT-SWAP",
    "SEI-USDT-SWAP",
    "STX-USDT-SWAP",
    "IMX-USDT-SWAP",
    "MANTA-USDT-SWAP",
    "MINA-USDT-SWAP",
    "VET-USDT-SWAP",
    "EOS-USDT-SWAP",
    "THETA-USDT-SWAP",
    "GALA-USDT-SWAP",
    "SAND-USDT-SWAP",
    "MANA-USDT-SWAP",
    "AXS-USDT-SWAP",
    "CRV-USDT-SWAP",
    "SNX-USDT-SWAP",
    "COMP-USDT-SWAP",
    "SUSHI-USDT-SWAP",
    "LDO-USDT-SWAP",
    "STORJ-USDT-SWAP",
    "APE-USDT-SWAP",
    "DYDX-USDT-SWAP",
    "ORDI-USDT-SWAP",
    "SATS-USDT-SWAP",
    "1000PEPE-USDT-SWAP",
    "1000SHIB-USDT-SWAP",
    "1000BONK-USDT-SWAP",
    "1000FLOKI-USDT-SWAP",
    "JASMY-USDT-SWAP",
    "CFX-USDT-SWAP",
    "ACH-USDT-SWAP",
    "SXP-USDT-SWAP",
    "GMT-USDT-SWAP",
    "ZIL-USDT-SWAP",
    "IOTA-USDT-SWAP",
    "XTZ-USDT-SWAP",
    "ALGO-USDT-SWAP",
    "ICP-USDT-SWAP",
    "HBAR-USDT-SWAP",
    "QNT-USDT-SWAP",
    "EGLD-USDT-SWAP",
    "FLOW-USDT-SWAP",
    "KSM-USDT-SWAP",
    "XLM-USDT-SWAP",
    "NEO-USDT-SWAP"
]


# ============================================================
# SESSION STATE
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "scan_time" not in st.session_state:
    st.session_state.scan_time = 0.0

if "universe_source" not in st.session_state:
    st.session_state.universe_source = "OKX"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt_price(value):
    if pd.isna(value):
        return "-"
    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:,.4f}"
    if value >= 0.01:
        return f"{value:.5f}"
    if value >= 0.0001:
        return f"{value:.7f}"
    return f"{value:.10f}"


def fmt_pct(value):
    if pd.isna(value):
        return "-"
    return f"{value:+.2f}%"


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


# ============================================================
# HTTP
# ============================================================

def okx_get(path, params=None, retries=2):
    url = BASE_URL + path

    last_error = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT
            )

            if response.status_code != 200:
                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:180]}"
                )

                if attempt < retries:
                    time.sleep(0.5)
                    continue

                return None, last_error

            payload = response.json()

            if payload.get("code") != "0":
                last_error = (
                    f"OKX code={payload.get('code')} "
                    f"msg={payload.get('msg')}"
                )

                if attempt < retries:
                    time.sleep(0.5)
                    continue

                return None, last_error

            return payload.get("data", []), None

        except Exception as exc:
            last_error = str(exc)

            if attempt < retries:
                time.sleep(0.5)
                continue

    return None, last_error


# ============================================================
# GET OKX UNIVERSE
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_okx_universe():
    """
    Ambil semua perpetual USDT yang sedang live
    dari endpoint ticker SWAP.

    Kita sengaja tidak bergantung pada instruments endpoint
    karena sebelumnya endpoint tersebut menghasilkan 403/451.
    """

    data, error = okx_get(
        "/api/v5/market/tickers",
        params={"instType": "SWAP"},
        retries=2
    )

    if data:
        symbols = []

        for item in data:
            inst_id = item.get("instId", "")
            state = item.get("instType", "")

            if (
                inst_id.endswith("-USDT-SWAP")
                and state == "SWAP"
            ):
                symbols.append(inst_id)

        symbols = sorted(list(set(symbols)))

        if symbols:
            return symbols, "OKX ticker"

    return FALLBACK_SYMBOLS.copy(), "Fallback static"


# ============================================================
# CANDLE DATA
# ============================================================

def get_candles(symbol, timeframe, limit=160):
    data, error = okx_get(
        "/api/v5/market/candles",
        params={
            "instId": symbol,
            "bar": timeframe,
            "limit": str(min(limit, 300))
        },
        retries=2
    )

    if not data:
        return None, error

    rows = []

    for row in data:
        if len(row) < 6:
            continue

        rows.append([
            safe_float(row[0]),
            safe_float(row[1]),
            safe_float(row[2]),
            safe_float(row[3]),
            safe_float(row[4]),
            safe_float(row[5]),
            row[8] if len(row) > 8 else "1"
        ])

    if not rows:
        return None, "No candle data"

    df = pd.DataFrame(
        rows,
        columns=[
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "confirm"
        ]
    )

    df["datetime"] = pd.to_datetime(
        df["ts"],
        unit="ms",
        utc=True
    )

    df = df.sort_values("ts").reset_index(drop=True)

    # Gunakan candle yang sudah confirm.
    df = df[df["confirm"] == "1"].copy()

    if len(df) < 60:
        return None, f"Insufficient candles: {len(df)}"

    return df, None


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    return result.fillna(50)


def atr(df, period=14):
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def zscore(series, period=30):
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()

    return (
        (series - mean) /
        std.replace(0, np.nan)
    ).fillna(0)


# ============================================================
# FEATURE ENGINE
# ============================================================

def calculate_features(df):
    df = df.copy()

    df["ema12"] = ema(df["close"], 12)
    df["ema24"] = ema(df["close"], 24)
    df["ema50"] = ema(df["close"], 50)

    df["rsi14"] = rsi(df["close"], 14)

    df["atr14"] = atr(df, 14)

    df["mom6"] = df["close"].pct_change(6) * 100
    df["mom12"] = df["close"].pct_change(12) * 100
    df["mom24"] = df["close"].pct_change(24) * 100

    df["ema12_dist"] = (
        (df["close"] / df["ema12"]) - 1
    ) * 100

    df["ema24_dist"] = (
        (df["close"] / df["ema24"]) - 1
    ) * 100

    df["volume_z"] = zscore(df["volume"], 30)

    df["trend_fast"] = (
        (df["ema12"] > df["ema24"]).astype(int) * 2 - 1
    )

    df["trend_slow"] = (
        (df["ema24"] > df["ema50"]).astype(int) * 2 - 1
    )

    return df


# ============================================================
# GPR ENGINE
# ============================================================

def calculate_gpr(df, horizon_candles):
    """
    GPR = directional ranking engine.

    Komponen:
    - momentum
    - trend EMA
    - RSI
    - volume
    - volatility
    - historical forward-return relationship

    Ini bukan machine-learning model murni.
    GPR digunakan sebagai composite ranking score.
    """

    if len(df) < 70:
        return None

    d = df.copy()

    last = d.iloc[-1]

    price = last["close"]
    atr_value = last["atr14"]

    if pd.isna(price) or price <= 0:
        return None

    if pd.isna(atr_value) or atr_value <= 0:
        atr_value = price * 0.005

    # --------------------------------------------------------
    # MOMENTUM COMPONENT
    # --------------------------------------------------------

    mom6 = safe_float(last["mom6"], 0)
    mom12 = safe_float(last["mom12"], 0)
    mom24 = safe_float(last["mom24"], 0)

    momentum_raw = (
        0.40 * mom6 +
        0.35 * mom12 +
        0.25 * mom24
    )

    momentum_component = np.tanh(
        momentum_raw / 2.5
    )

    # --------------------------------------------------------
    # TREND COMPONENT
    # --------------------------------------------------------

    ema12_dist = safe_float(
        last["ema12_dist"],
        0
    )

    ema24_dist = safe_float(
        last["ema24_dist"],
        0
    )

    trend_component = np.tanh(
        (
            0.55 * ema12_dist +
            0.45 * ema24_dist
        ) / 1.5
    )

    # --------------------------------------------------------
    # RSI COMPONENT
    # --------------------------------------------------------

    rsi_value = safe_float(
        last["rsi14"],
        50
    )

    # RSI 50 = neutral
    # 55-65 = bullish zone
    # 35-45 = bearish zone
    # extreme RSI dikurangi agar tidak mengejar overextension

    rsi_component = clamp(
        (rsi_value - 50) / 20,
        -1,
        1
    )

    if rsi_value > 75:
        rsi_component *= 0.55

    if rsi_value < 25:
        rsi_component *= 0.55

    # --------------------------------------------------------
    # VOLUME COMPONENT
    # --------------------------------------------------------

    volume_z = safe_float(
        last["volume_z"],
        0
    )

    volume_component = np.tanh(
        volume_z / 2.0
    )

    # --------------------------------------------------------
    # VOLATILITY COMPONENT
    # --------------------------------------------------------

    atr_pct = (
        atr_value / price
    ) * 100

    # terlalu volatil = confidence turun
    volatility_penalty = clamp(
        (atr_pct - 1.5) / 3.0,
        0,
        1
    )

    # --------------------------------------------------------
    # COMBINED DIRECTION SCORE
    # --------------------------------------------------------

    directional_score = (
        0.40 * momentum_component +
        0.30 * trend_component +
        0.15 * rsi_component +
        0.15 * volume_component
    )

    # --------------------------------------------------------
    # FUTURE RETURN MODEL
    # --------------------------------------------------------

    # Forecast horizon.
    # Kita gunakan relationship historis sederhana:
    # current feature -> future return.

    future_return = (
        d["close"].shift(-horizon_candles) /
        d["close"] - 1
    ) * 100

    valid = pd.DataFrame({
        "direction": (
            0.40 * np.tanh(
                d["mom6"] / 2.5
            ) +
            0.30 * np.tanh(
                (
                    0.55 * d["ema12_dist"] +
                    0.45 * d["ema24_dist"]
                ) / 1.5
            ) +
            0.15 * ((d["rsi14"] - 50) / 20) +
            0.15 * np.tanh(
                d["volume_z"] / 2.0
            )
        ),
        "future": future_return
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid) >= 30:
        recent = valid.tail(120)

        direction_arr = recent["direction"].values
        future_arr = recent["future"].values

        if (
            np.std(direction_arr) > 0 and
            np.std(future_arr) > 0
        ):
            correlation = np.corrcoef(
                direction_arr,
                future_arr
            )[0, 1]
        else:
            correlation = 0.0
    else:
        correlation = 0.0

    if not np.isfinite(correlation):
        correlation = 0.0

    correlation = clamp(
        correlation,
        -1,
        1
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    # ATR based expected movement.
    expected_move = (
        atr_pct *
        max(1.0, np.sqrt(horizon_candles / 4))
    )

    forecast_strength = (
        0.60 * abs(directional_score) +
        0.25 * abs(correlation) +
        0.15 * clamp(
            abs(momentum_raw) / 5,
            0,
            1
        )
    )

    forecast_strength = clamp(
        forecast_strength,
        0,
        1
    )

    forecast = (
        np.sign(directional_score) *
        expected_move *
        forecast_strength
    )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    recent_returns = d["close"].pct_change().tail(30)

    realized_vol = (
        recent_returns.std() *
        100
    )

    if not np.isfinite(realized_vol):
        realized_vol = atr_pct

    uncertainty = (
        0.60 * realized_vol +
        0.40 * atr_pct
    )

    # correlation lemah -> uncertainty naik
    uncertainty *= (
        1.0 +
        0.40 * (1 - abs(correlation))
    )

    uncertainty = max(
        0.05,
        float(uncertainty)
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    # Signal agreement
    components = np.array([
        momentum_component,
        trend_component,
        rsi_component,
        volume_component
    ])

    direction_sign = np.sign(
        directional_score
    )

    if direction_sign == 0:
        agreement = 0.5
    else:
        same_direction = (
            np.sign(components) == direction_sign
        )

        agreement = float(
            np.mean(same_direction)
        )

    # Strength 0-1
    strength = abs(directional_score)

    # Historical validation
    validation = abs(correlation)

    # Risk from uncertainty
    uncertainty_ratio = (
        abs(forecast) /
        max(uncertainty, 0.05)
    )

    risk_quality = clamp(
        uncertainty_ratio / 4.0,
        0,
        1
    )

    confidence = (
        0.35 * strength +
        0.25 * agreement +
        0.25 * validation +
        0.15 * risk_quality
    )

    confidence_pct = clamp(
        confidence * 100,
        0,
        100
    )

    # --------------------------------------------------------
    # GPR SCORE
    # --------------------------------------------------------

    # Score dibuat 0-100.
    # 50 = neutral.
    # >50 bullish bias
    # <50 bearish bias

    base_score = (
        50 +
        directional_score * 50
    )

    validation_boost = (
        correlation *
        10 *
        strength
    )

    gpr_score = clamp(
        base_score + validation_boost,
        0,
        100
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    if gpr_score >= 50:
        signal = "LONG"
    else:
        signal = "SHORT"

    # --------------------------------------------------------
    # CONFIDENCE LABEL
    # --------------------------------------------------------

    if confidence_pct >= 85:
        confidence_label = "SANGAT KUAT"
    elif confidence_pct >= 70:
        confidence_label = "KUAT"
    elif confidence_pct >= 55:
        confidence_label = "MODERAT"
    else:
        confidence_label = "LEMAH"

    # --------------------------------------------------------
    # ATR TP / SL
    # --------------------------------------------------------

    if signal == "LONG":
        stop_loss = price - (
            1.5 * atr_value
        )

        take_profit = price + (
            2.5 * atr_value
        )

    else:
        stop_loss = price + (
            1.5 * atr_value
        )

        take_profit = price - (
            2.5 * atr_value
        )

    risk = abs(
        price - stop_loss
    )

    reward = abs(
        take_profit - price
    )

    risk_reward = (
        reward / risk
        if risk > 0 else np.nan
    )

    # --------------------------------------------------------
    # SIGNAL QUALITY
    # --------------------------------------------------------

    quality_score = (
        0.55 * gpr_score +
        0.45 * confidence_pct
    )

    return {
        "signal": signal,
        "gpr_score": float(gpr_score),
        "gpr_return": float(forecast),
        "uncertainty": float(uncertainty),
        "confidence": float(confidence_pct),
        "confidence_label": confidence_label,
        "price": float(price),
        "rsi": float(rsi_value),
        "mom_6h": float(mom6),
        "mom_12h": float(mom12),
        "mom_24h": float(mom24),
        "volume_z": float(volume_z),
        "ema12_dist": float(ema12_dist),
        "ema24_dist": float(ema24_dist),
        "atr": float(atr_value),
        "atr_pct": float(atr_pct),
        "forecast": float(forecast),
        "correlation": float(correlation),
        "agreement": float(agreement * 100),
        "quality_score": float(quality_score),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "risk_reward": float(risk_reward),
        "candle_time": last["datetime"]
    }


# ============================================================
# MULTI TIMEFRAME CONFIRMATION
# ============================================================

def get_1h_confirmation(symbol):
    """
    Konfirmasi trend 1H.

    Return:
    +1 bullish
    -1 bearish
     0 neutral/error
    """

    df, error = get_candles(
        symbol,
        "1H",
        limit=100
    )

    if df is None:
        return 0, error

    df = calculate_features(df)

    last = df.iloc[-1]

    ema12 = last["ema12"]
    ema24 = last["ema24"]
    ema50 = last["ema50"]

    if (
        ema12 > ema24 and
        ema24 > ema50
    ):
        return 1, None

    if (
        ema12 < ema24 and
        ema24 < ema50
    ):
        return -1, None

    return 0, None


# ============================================================
# SCAN ONE SYMBOL
# ============================================================

def scan_symbol(symbol, timeframe, horizon):
    try:
        df, error = get_candles(
            symbol,
            timeframe,
            limit=180
        )

        if df is None:
            return None, {
                "symbol": symbol,
                "status": error or "No data"
            }

        df = calculate_features(df)

        result = calculate_gpr(
            df,
            horizon
        )

        if result is None:
            return None, {
                "symbol": symbol,
                "status": "Failed GPR calculation"
            }

        # Multi-timeframe confirmation
        trend_1h, trend_error = get_1h_confirmation(
            symbol
        )

        result["trend_1h"] = trend_1h

        if trend_1h == 1:
            result["trend_1h_label"] = "BULLISH"
        elif trend_1h == -1:
            result["trend_1h_label"] = "BEARISH"
        else:
            result["trend_1h_label"] = "NEUTRAL"

        # ----------------------------------------------------
        # MTF BONUS / PENALTY
        # ----------------------------------------------------

        if result["signal"] == "LONG":
            if trend_1h == 1:
                result["mtf_alignment"] = "ALIGNED"
                result["quality_score"] += 7
            elif trend_1h == -1:
                result["mtf_alignment"] = "CONFLICT"
                result["quality_score"] -= 8
            else:
                result["mtf_alignment"] = "NEUTRAL"

        else:
            if trend_1h == -1:
                result["mtf_alignment"] = "ALIGNED"
                result["quality_score"] += 7
            elif trend_1h == 1:
                result["mtf_alignment"] = "CONFLICT"
                result["quality_score"] -= 8
            else:
                result["mtf_alignment"] = "NEUTRAL"

        result["quality_score"] = clamp(
            result["quality_score"]
        )

        return {
            "symbol": symbol,
            **result
        }, None

    except Exception as exc:
        return None, {
            "symbol": symbol,
            "status": str(exc)
        }


# ============================================================
# SCAN ALL
# ============================================================

def run_scan(
    symbols,
    timeframe,
    horizon,
    workers
):
    results = []
    errors = []

    start = time.time()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                scan_symbol,
                symbol,
                timeframe,
                horizon
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            result, error = future.result()

            if result is not None:
                results.append(result)

            if error is not None:
                errors.append(error)

    elapsed = time.time() - start

    if not results:
        return pd.DataFrame(), errors, elapsed

    df = pd.DataFrame(results)

    df = df.sort_values(
        by=[
            "quality_score",
            "gpr_score",
            "confidence"
        ],
        ascending=False
    ).reset_index(drop=True)

    df.insert(
        0,
        "rank",
        range(1, len(df) + 1)
    )

    return df, errors, elapsed


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Scanner Settings")

    timeframe = st.selectbox(
        "Timeframe",
        options=[
            "5m",
            "15m",
            "30m",
            "1H",
            "4H"
        ],
        index=1
    )

    horizon_options = {
        "5 candle": 5,
        "10 candle": 10,
        "20 candle": 20,
        "30 candle": 30,
        "40 candle": 40
    }

    horizon_label = st.selectbox(
        "Horizon GPR",
        options=list(
            horizon_options.keys()
        ),
        index=1
    )

    horizon = horizon_options[
        horizon_label
    ]

    coin_count = st.selectbox(
        "Jumlah coin",
        options=[
            20,
            30,
            50,
            79,
            100
        ],
        index=3
    )

    minimum_score = st.slider(
        "Minimum GPR Score",
        min_value=50,
        max_value=95,
        value=70,
        step=1
    )

    minimum_confidence = st.slider(
        "Minimum Confidence",
        min_value=0,
        max_value=95,
        value=60,
        step=5
    )

    workers = st.slider(
        "Parallel workers",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    st.divider()

    scan_button = st.button(
        "🚀 SCAN FUTURES",
        type="primary",
        use_container_width=True
    )

    clear_button = st.button(
        "🗑️ CLEAR CACHE",
        use_container_width=True
    )

    st.caption(
        "Public market data • OKX SWAP • No API key"
    )


# ============================================================
# CLEAR CACHE
# ============================================================

if clear_button:
    st.cache_data.clear()
    st.session_state.scan_results = None
    st.session_state.scan_errors = []
    st.session_state.scan_time = 0.0
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("📈 GPR CRYPTO FUTURES SCANNER")

st.caption(
    f"Target utama: USDT Perpetual • "
    f"Timeframe {timeframe} • "
    f"GPR sebagai mesin ranking utama"
)

st.info(
    f"Mode data: public market data, tanpa API key. "
    f"Timeframe {timeframe} • "
    f"Horizon {horizon} candle "
    f"({horizon * {'5m':5, '15m':15, '30m':30, '1H':60, '4H':240}[timeframe] / 60:.2f} jam)."
)


# ============================================================
# GET UNIVERSE
# ============================================================

symbols, universe_source = get_okx_universe()

# Ambil hanya sejumlah coin sesuai setting
symbols = symbols[:coin_count]

st.session_state.universe_source = universe_source


# ============================================================
# SCAN
# ============================================================

if scan_button:

    progress = st.progress(
        0,
        text="Menyiapkan scanner..."
    )

    # run scan
    result_df, errors, elapsed = run_scan(
        symbols,
        timeframe,
        horizon,
        workers
    )

    progress.progress(
        100,
        text="Scan selesai."
    )

    st.session_state.scan_results = result_df
    st.session_state.scan_errors = errors
    st.session_state.scan_time = elapsed

    st.rerun()


# ============================================================
# NO SCAN YET
# ============================================================

if st.session_state.scan_results is None:

    st.warning(
        "Klik **🚀 SCAN FUTURES** untuk menjalankan scanner."
    )

    st.stop()


# ============================================================
# RESULT
# ============================================================

df = st.session_state.scan_results
errors = st.session_state.scan_errors
elapsed = st.session_state.scan_time


# ============================================================
# FILTER
# ============================================================

if not df.empty:

    candidates = df[
        (df["gpr_score"] >= minimum_score) &
        (df["confidence"] >= minimum_confidence)
    ].copy()

else:
    candidates = pd.DataFrame()


# ============================================================
# STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Universe",
        len(symbols)
    )

with col2:
    st.metric(
        "Berhasil dianalisis",
        len(df)
    )

with col3:
    st.metric(
        "Gagal",
        len(errors)
    )

st.success(
    f"Scan selesai: {len(df)} berhasil, "
    f"{len(errors)} gagal dalam {elapsed:.1f} detik."
)


# ============================================================
# SUMMARY
# ============================================================

if not df.empty:

    top_row = df.iloc[0]

    if candidates.empty:
        best_return = df["gpr_return"].iloc[0]
        best_score = df["gpr_score"].iloc[0]
    else:
        best_return = candidates["gpr_return"].iloc[0]
        best_score = candidates["gpr_score"].iloc[0]

else:
    best_return = np.nan
    best_score = np.nan


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Lolos GPR Score",
        len(
            df[
                df["gpr_score"] >= minimum_score
            ]
        ) if not df.empty else 0
    )

with c2:
    st.metric(
        "Kandidat Trading",
        len(candidates)
    )

with c3:
    st.metric(
        "GPR Return Tertinggi",
        fmt_pct(best_return)
    )

with c4:
    st.metric(
        "Score Tertinggi",
        f"{best_score:.1f}"
        if np.isfinite(best_score)
        else "-"
    )


# ============================================================
# RANKING
# ============================================================

st.divider()

st.subheader("🏆 GPR Futures Ranking")

if df.empty:

    st.error(
        "Tidak ada data berhasil dianalisis."
    )

else:

    ranking = df.copy()

    ranking["COIN"] = ranking["symbol"].str.replace(
        "-USDT-SWAP",
        "",
        regex=False
    )

    ranking["SIGNAL"] = ranking["signal"]

    ranking["GPR SCORE"] = ranking[
        "gpr_score"
    ].round(1)

    ranking["GPR RETURN"] = ranking[
        "gpr_return"
    ].map(fmt_pct)

    ranking["UNCERTAINTY"] = (
        ranking["uncertainty"]
        .map(lambda x: f"{x:.2f}%")
    )

    ranking["CONFIDENCE"] = (
        ranking["confidence"]
        .map(lambda x: f"{x:.0f}%")
    )

    ranking["CONF LABEL"] = ranking[
        "confidence_label"
    ]

    ranking["PRICE"] = ranking[
        "price"
    ].map(fmt_price)

    ranking["RSI"] = ranking[
        "rsi"
    ].round(1)

    ranking["MOM 6H"] = ranking[
        "mom_6h"
    ].map(fmt_pct)

    ranking["MOM 12H"] = ranking[
        "mom_12h"
    ].map(fmt_pct)

    ranking["MOM 24H"] = ranking[
        "mom_24h"
    ].map(fmt_pct)

    ranking["VOL Z"] = ranking[
        "volume_z"
    ].round(2)

    ranking["MTF"] = ranking[
        "mtf_alignment"
    ]

    ranking["QUALITY"] = ranking[
        "quality_score"
    ].round(1)

    display_columns = [
        "rank",
        "COIN",
        "SIGNAL",
        "GPR SCORE",
        "GPR RETURN",
        "UNCERTAINTY",
        "CONFIDENCE",
        "CONF LABEL",
        "MTF",
        "QUALITY",
        "PRICE",
        "RSI",
        "MOM 6H",
        "MOM 12H",
        "MOM 24H",
        "VOL Z"
    ]

    st.dataframe(
        ranking[display_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TRADING CANDIDATES
# ============================================================

st.divider()

st.subheader("⭐ Kandidat Trading")

if candidates.empty:

    st.warning(
        f"Belum ada kandidat yang memenuhi "
        f"GPR Score ≥ {minimum_score} dan "
        f"Confidence ≥ {minimum_confidence}%."
    )

    st.caption(
        "Turunkan threshold jika ingin melihat "
        "lebih banyak kandidat."
    )

else:

    candidate_cols = st.columns(
        min(3, len(candidates))
    )

    for i, (_, row) in enumerate(
        candidates.head(6).iterrows()
    ):

        with candidate_cols[
            i % len(candidate_cols)
        ]:

            symbol_name = row[
                "symbol"
            ].replace(
                "-USDT-SWAP",
                ""
            )

            if row["signal"] == "LONG":
                signal_icon = "🟢"
            else:
                signal_icon = "🔴"

            st.markdown(
                f"### {symbol_name}"
            )

            st.write(
                f"{signal_icon} **{row['signal']}**"
            )

            st.metric(
                "GPR Score",
                f"{row['gpr_score']:.1f}"
            )

            st.metric(
                "Confidence",
                f"{row['confidence']:.0f}%"
            )

            st.metric(
                "Forecast",
                fmt_pct(row["gpr_return"])
            )

            st.write(
                f"**Level:** {row['confidence_label']}"
            )

            st.write(
                f"**MTF 1H:** {row['trend_1h_label']}"
            )

            st.write(
                f"**Risk/Reward:** "
                f"{row['risk_reward']:.2f}"
            )


# ============================================================
# CONFIDENCE EXPLANATION
# ============================================================

st.divider()

st.subheader("🧠 Cara Membaca Confidence")

st.info(
    "Confidence bukan probabilitas bahwa trade pasti profit. "
    "Confidence adalah ukuran seberapa kuat dan konsisten "
    "komponen sinyal mendukung arah yang dipilih."
)

confidence_table = pd.DataFrame({
    "Confidence": [
        "85–100%",
        "70–84%",
        "55–69%",
        "<55%"
    ],
    "Interpretasi": [
        "SANGAT KUAT — sinyal sangat konsisten",
        "KUAT — cukup solid untuk kandidat utama",
        "MODERAT — sinyal ada tetapi perlu konfirmasi",
        "LEMAH — jangan jadikan kandidat utama"
    ]
})

st.dataframe(
    confidence_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DETAIL ANALYSIS
# ============================================================

st.divider()

st.subheader("🔎 Detail Analysis")

if not df.empty:

    detail_symbols = [
        s.replace("-USDT-SWAP", "")
        for s in df["symbol"].tolist()
    ]

    selected_coin = st.selectbox(
        "Pilih coin",
        options=detail_symbols
    )

    selected_symbol = (
        selected_coin +
        "-USDT-SWAP"
    )

    selected = df[
        df["symbol"] == selected_symbol
    ].iloc[0]

    # --------------------------------------------------------
    # MAIN SIGNAL
    # --------------------------------------------------------

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        st.metric(
            "SIGNAL",
            selected["signal"]
        )

    with a2:
        st.metric(
            "GPR SCORE",
            f"{selected['gpr_score']:.1f}"
        )

    with a3:
        st.metric(
            "FORECAST",
            fmt_pct(
                selected["gpr_return"]
            )
        )

    with a4:
        st.metric(
            "CONFIDENCE",
            f"{selected['confidence']:.0f}%"
        )

    # --------------------------------------------------------
    # CONFIDENCE LEVEL
    # --------------------------------------------------------

    confidence = selected[
        "confidence"
    ]

    if confidence >= 85:
        st.success(
            f"Confidence {confidence:.0f}% — "
            f"SANGAT KUAT"
        )
    elif confidence >= 70:
        st.success(
            f"Confidence {confidence:.0f}% — "
            f"KUAT"
        )
    elif confidence >= 55:
        st.warning(
            f"Confidence {confidence:.0f}% — "
            f"MODERAT"
        )
    else:
        st.error(
            f"Confidence {confidence:.0f}% — "
            f"LEMAH"
        )

    # --------------------------------------------------------
    # TECHNICAL DATA
    # --------------------------------------------------------

    st.markdown("### 📊 Technical Data")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.metric(
            "Harga",
            fmt_price(
                selected["price"]
            )
        )

    with t2:
        st.metric(
            "RSI 14",
            f"{selected['rsi']:.1f}"
        )

    with t3:
        st.metric(
            "Momentum 6H",
            fmt_pct(
                selected["mom_6h"]
            )
        )

    with t4:
        st.metric(
            "Momentum 12H",
            fmt_pct(
                selected["mom_12h"]
            )
        )

    t5, t6, t7, t8 = st.columns(4)

    with t5:
        st.metric(
            "Momentum 24H",
            fmt_pct(
                selected["mom_24h"]
            )
        )

    with t6:
        st.metric(
            "Volume Z",
            f"{selected['volume_z']:.2f}"
        )

    with t7:
        st.metric(
            "EMA12 Distance",
            fmt_pct(
                selected["ema12_dist"]
            )
        )

    with t8:
        st.metric(
            "EMA24 Distance",
            fmt_pct(
                selected["ema24_dist"]
            )
        )

    # --------------------------------------------------------
    # RISK MANAGEMENT
    # --------------------------------------------------------

    st.markdown("### 🎯 Risk Management")

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Entry / Current",
            fmt_price(
                selected["price"]
            )
        )

    with r2:
        st.metric(
            "Stop Loss",
            fmt_price(
                selected["stop_loss"]
            )
        )

    with r3:
        st.metric(
            "Take Profit",
            fmt_price(
                selected["take_profit"]
            )
        )

    with r4:
        st.metric(
            "Risk / Reward",
            f"1 : {selected['risk_reward']:.2f}"
        )

    # --------------------------------------------------------
    # SIGNAL COMPONENTS
    # --------------------------------------------------------

    st.markdown("### 🔬 GPR Signal Components")

    component_df = pd.DataFrame({
        "Component": [
            "Momentum",
            "Trend EMA",
            "RSI",
            "Volume",
            "Historical validation",
            "Signal agreement",
            "1H confirmation"
        ],
        "Value": [
            selected["mom_6h"],
            selected["ema12_dist"],
            selected["rsi"],
            selected["volume_z"],
            selected["correlation"],
            selected["agreement"],
            selected["trend_1h_label"]
        ]
    })

    st.dataframe(
        component_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    st.markdown("### 📝 Interpretasi")

    signal = selected["signal"]
    score = selected["gpr_score"]
    conf = selected["confidence"]
    mtf = selected["mtf_alignment"]

    if signal == "LONG":
        direction_text = (
            "bias utama saat ini adalah **LONG**."
        )
    else:
        direction_text = (
            "bias utama saat ini adalah **SHORT**."
        )

    if mtf == "ALIGNED":
        mtf_text = (
            "Trend 1H searah dengan sinyal sehingga "
            "memberikan konfirmasi tambahan."
        )
    elif mtf == "CONFLICT":
        mtf_text = (
            "Trend 1H berlawanan dengan sinyal 15m. "
            "Ini berarti perlu lebih berhati-hati."
        )
    else:
        mtf_text = (
            "Trend 1H belum memberikan konfirmasi kuat."
        )

    st.write(
        f"**{selected_coin}** memiliki GPR Score "
        f"**{score:.1f}** dan Confidence "
        f"**{conf:.0f}%**. Berdasarkan kombinasi "
        f"momentum, trend, RSI, volume dan validasi "
        f"historis, {direction_text.capitalize()}"
    )

    st.write(mtf_text)


# ============================================================
# ERROR / DIAGNOSTIC
# ============================================================

st.divider()

with st.expander(
    f"📋 ERROR / DIAGNOSTIC ({len(errors)})"
):

    if not errors:

        st.success(
            "Tidak ada error."
        )

    else:

        diagnostic_rows = []

        for item in errors:

            diagnostic_rows.append({
                "Symbol": item.get(
                    "symbol",
                    "-"
                ),
                "Status": item.get(
                    "status",
                    "Unknown error"
                )
            })

        diagnostic_df = pd.DataFrame(
            diagnostic_rows
        )

        st.dataframe(
            diagnostic_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GPR Crypto Futures Scanner • "
    "USDT Perpetual • "
    f"{timeframe} • "
    "OKX Public Market Data • "
    "Research Prototype"
)
