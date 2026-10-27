import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="GPR Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

OKX_BASE_URL = "https://www.okx.com"

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 2.1rem !important;
}

h2 {
    font-size: 1.55rem !important;
}

h3 {
    font-size: 1.25rem !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.65rem;
}

.signal-long {
    background: #dff7e8;
    color: #087f3e;
    padding: 5px 10px;
    border-radius: 12px;
    font-weight: 700;
}

.signal-short {
    background: #ffe2e5;
    color: #c62828;
    padding: 5px 10px;
    border-radius: 12px;
    font-weight: 700;
}

.conf-strong {
    background: #dff7e8;
    color: #087f3e;
    padding: 5px 10px;
    border-radius: 12px;
    font-weight: 700;
}

.conf-medium {
    background: #fff3cd;
    color: #946200;
    padding: 5px 10px;
    border-radius: 12px;
    font-weight: 700;
}

.conf-weak {
    background: #ffe2e5;
    color: #b42318;
    padding: 5px 10px;
    border-radius: 12px;
    font-weight: 700;
}

.small-note {
    color: #7b8190;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

TIMEFRAME_MAP = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "1H",
    "4H": "4H",
}

HORIZON_MAP = {
    "5 candle / 1.25 jam": 5,
    "10 candle / 2.5 jam": 10,
    "20 candle / 5 jam": 20,
    "30 candle / 7.5 jam": 30,
    "40 candle / 10 jam": 40,
}

DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK",
    "DOT", "LTC", "BCH", "UNI", "AAVE", "NEAR", "APT", "SUI",
    "WLD", "ARB", "OP", "ATOM", "FIL", "ETC", "ICP", "INJ",
    "HBAR", "MKR", "TON", "TRX", "XLM", "ALGO", "EOS", "RUNE",
    "KAS", "JUP", "TIA", "SEI", "PEPE", "SHIB", "BONK", "FLOKI",
    "WIF", "STX", "IMX", "GRT", "SAND", "MANA", "AXS", "CRV",
    "DYDX", "SNX", "COMP", "MANTA", "JTO", "PYTH", "ENA", "EIGEN",
    "STRK", "ZRO", "BLUR", "ORDI", "SATS", "NOT", "MEME", "MINA",
    "TAO", "JASMY", "ONDO", "PENDLE", "RENDER", "FET", "APE",
    "GMT", "MASK", "1INCH", "LDO", "ENS"
]


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "GPR-Crypto-Futures-Scanner/1.0"
    })
    return session


SESSION = get_session()


# ============================================================
# OKX API
# ============================================================

def okx_get(endpoint, params=None, timeout=15):
    try:
        url = OKX_BASE_URL + endpoint

        response = SESSION.get(
            url,
            params=params,
            timeout=timeout
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("code") != "0":
            raise RuntimeError(
                f"OKX code={payload.get('code')} "
                f"msg={payload.get('msg')}"
            )

        return payload.get("data", [])

    except Exception as exc:
        raise RuntimeError(str(exc))


# ============================================================
# GET REAL SWAP INSTRUMENTS
# ============================================================

@st.cache_data(ttl=300)
def get_swap_instruments():
    """
    Mengambil daftar perpetual USDT dari OKX.
    Ini mencegah error 51001 akibat symbol yang sudah tidak tersedia.
    """

    data = okx_get(
        "/api/v5/public/instruments",
        params={
            "instType": "SWAP"
        }
    )

    instruments = []

    for item in data:

        inst_id = item.get("instId", "")

        if not inst_id.endswith("-USDT-SWAP"):
            continue

        state = item.get("state", "")

        if state != "live":
            continue

        base = item.get("baseCcy", "")

        if not base:
            continue

        instruments.append({
            "base": base,
            "instId": inst_id,
            "tickSz": item.get("tickSz", ""),
            "lotSz": item.get("lotSz", ""),
        })

    instruments = sorted(
        instruments,
        key=lambda x: x["base"]
    )

    return instruments


# ============================================================
# CANDLE DATA
# ============================================================

def get_candles(inst_id, bar="15m", limit=200):

    data = okx_get(
        "/api/v5/market/candles",
        params={
            "instId": inst_id,
            "bar": bar,
            "limit": str(min(limit, 300))
        }
    )

    if not data:
        raise RuntimeError("No candle data")

    rows = []

    for x in data:

        rows.append([
            int(x[0]),
            float(x[1]),
            float(x[2]),
            float(x[3]),
            float(x[4]),
            float(x[5]),
            float(x[6]),
            float(x[7]),
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volCcy",
            "volCcyQuote",
        ]
    )

    df = df.sort_values("ts").reset_index(drop=True)

    # Hanya candle yang sudah confirmed.
    # OKX candle memiliki flag pada index 8 jika tersedia.
    # Jika field tidak tersedia, kita tetap menggunakan data.
    if len(data[0]) >= 9:
        rows2 = []

        for x in data:
            rows2.append(int(x[8]))

        df["confirm"] = rows2
        df = df[df["confirm"] == 1].copy()

    return df.reset_index(drop=True)


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


def zscore(series, window=20):

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    z = (series - mean) / std.replace(0, np.nan)

    return z.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(df):

    if len(df) < 60:
        raise RuntimeError("Candle tidak cukup")

    df = df.copy()

    close = df["close"]
    volume = df["volume"]

    df["ema12"] = ema(close, 12)
    df["ema24"] = ema(close, 24)
    df["rsi14"] = rsi(close, 14)

    df["mom6"] = close.pct_change(6) * 100
    df["mom12"] = close.pct_change(12) * 100
    df["mom24"] = close.pct_change(24) * 100

    df["ema12_dist"] = (
        (close - df["ema12"])
        / df["ema12"]
    ) * 100

    df["ema24_dist"] = (
        (close - df["ema24"])
        / df["ema24"]
    ) * 100

    df["vol_z"] = zscore(volume, 20)

    return df


# ============================================================
# HISTORICAL VALIDATION
# ============================================================

def historical_validation(df, horizon):

    close = df["close"]

    future_return = (
        close.shift(-horizon) / close - 1
    ) * 100

    signal_return = []

    for i in range(len(df) - horizon):

        row = df.iloc[i]

        bullish = (
            row["ema12_dist"] > 0
            and row["mom12"] > 0
        )

        bearish = (
            row["ema12_dist"] < 0
            and row["mom12"] < 0
        )

        if bullish:
            signal_return.append(
                1 if future_return.iloc[i] > 0 else 0
            )

        elif bearish:
            signal_return.append(
                1 if future_return.iloc[i] < 0 else 0
            )

    if not signal_return:
        return 0.50

    return float(np.mean(signal_return))


# ============================================================
# GPR SCORING
# ============================================================

def calculate_gpr(df, horizon):

    row = df.iloc[-1]

    # --------------------------------------------------------
    # NORMALIZED COMPONENTS
    # --------------------------------------------------------

    # Momentum
    momentum_raw = (
        0.45 * row["mom6"] +
        0.35 * row["mom12"] +
        0.20 * row["mom24"]
    )

    momentum_score = np.tanh(
        momentum_raw / 2.5
    )

    # Trend EMA
    trend_raw = (
        0.60 * row["ema12_dist"] +
        0.40 * row["ema24_dist"]
    )

    trend_score = np.tanh(
        trend_raw / 1.0
    )

    # RSI
    rsi_score = np.clip(
        (row["rsi14"] - 50) / 20,
        -1,
        1
    )

    # Volume
    volume_score = np.clip(
        row["vol_z"] / 2.5,
        -1,
        1
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    directional_score = (
        0.35 * momentum_score +
        0.30 * trend_score +
        0.20 * rsi_score +
        0.15 * volume_score
    )

    signal = (
        "LONG"
        if directional_score >= 0
        else "SHORT"
    )

    # --------------------------------------------------------
    # COMPONENT AGREEMENT
    # --------------------------------------------------------

    components = [
        np.sign(momentum_score),
        np.sign(trend_score),
        np.sign(rsi_score),
        np.sign(volume_score),
    ]

    signal_sign = 1 if signal == "LONG" else -1

    agreement = sum(
        1 for x in components
        if x == signal_sign
    ) / len(components)

    # --------------------------------------------------------
    # HISTORICAL VALIDATION
    # --------------------------------------------------------

    hist_validation = historical_validation(
        df,
        horizon
    )

    # --------------------------------------------------------
    # RAW SCORE
    # --------------------------------------------------------

    raw_strength = abs(directional_score)

    # Convert to 0-100.
    #
    # Important:
    # Score is NOT probability.
    # It is model strength.
    #
    score = (
        50 +
        raw_strength * 45
    )

    # Agreement bonus
    score += (
        agreement - 0.50
    ) * 10

    # Historical validation bonus
    score += (
        hist_validation - 0.50
    ) * 15

    score = float(
        np.clip(score, 0, 100)
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    # Confidence is deliberately bounded.
    #
    # It combines:
    # 40% directional strength
    # 25% component agreement
    # 20% historical validation
    # 15% trend consistency
    #

    trend_consistency = (
        1.0
        if np.sign(row["ema12_dist"])
        == np.sign(row["ema24_dist"])
        else 0.0
    )

    confidence = (
        0.40 * raw_strength +
        0.25 * agreement +
        0.20 * hist_validation +
        0.15 * trend_consistency
    )

    confidence_pct = float(
        np.clip(
            confidence * 100,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    uncertainty = (
        100 - confidence_pct
    )

    # scale to more useful percentage
    uncertainty = (
        0.25 +
        uncertainty * 0.025
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    volatility = (
        df["close"]
        .pct_change()
        .rolling(20)
        .std()
        .iloc[-1]
        * 100
    )

    if not np.isfinite(volatility):
        volatility = 1.0

    forecast_strength = (
        0.70 * abs(row["mom12"]) +
        0.30 * abs(row["mom24"])
    )

    forecast = (
        forecast_strength *
        (0.65 + confidence_pct / 200)
    )

    forecast = max(
        forecast,
        volatility * 0.50
    )

    forecast = float(
        min(forecast, 15.0)
    )

    if signal == "SHORT":
        forecast = -forecast

    # --------------------------------------------------------
    # CONFIDENCE LABEL
    # --------------------------------------------------------

    if confidence_pct >= 80:
        confidence_label = "SANGAT KUAT"
    elif confidence_pct >= 65:
        confidence_label = "KUAT"
    elif confidence_pct >= 50:
        confidence_label = "SEDANG"
    elif confidence_pct >= 35:
        confidence_label = "LEMAH"
    else:
        confidence_label = "SANGAT LEMAH"

    return {
        "signal": signal,
        "score": score,
        "forecast": forecast,
        "uncertainty": uncertainty,
        "confidence": confidence_pct,
        "confidence_label": confidence_label,
        "agreement": agreement * 100,
        "historical_validation": hist_validation * 100,
        "momentum_component": momentum_score,
        "trend_component": trend_score,
        "rsi_component": rsi_score,
        "volume_component": volume_score,
        "trend_consistency": trend_consistency,
        "price": row["close"],
        "rsi": row["rsi14"],
        "mom6": row["mom6"],
        "mom12": row["mom12"],
        "mom24": row["mom24"],
        "vol_z": row["vol_z"],
        "ema12_dist": row["ema12_dist"],
        "ema24_dist": row["ema24_dist"],
        "timestamp": int(df["ts"].iloc[-1]),
    }


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_coin(inst, bar, horizon):

    base = inst["base"]
    inst_id = inst["instId"]

    try:

        candles = get_candles(
            inst_id,
            bar=bar,
            limit=200
        )

        if len(candles) < 60:
            raise RuntimeError(
                f"Candle hanya {len(candles)}"
            )

        df = calculate_features(candles)

        result = calculate_gpr(
            df,
            horizon
        )

        result["coin"] = base
        result["inst_id"] = inst_id
        result["status"] = "OK"

        return result

    except Exception as exc:

        return {
            "coin": base,
            "inst_id": inst_id,
            "status": "ERROR",
            "error": str(exc),
        }


# ============================================================
# SCAN
# ============================================================

def scan_market(instruments, bar, horizon, workers):

    start = time.time()

    results = []
    errors = []

    selected = instruments

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                analyze_coin,
                inst,
                bar,
                horizon
            ): inst
            for inst in selected
        }

        for future in as_completed(futures):

            result = future.result()

            if result.get("status") == "OK":
                results.append(result)
            else:
                errors.append(result)

    # ========================================================
    # CRITICAL:
    # DETERMINISTIC SORTING
    # ========================================================
    #
    # Primary:
    #   GPR score descending
    #
    # Secondary:
    #   confidence descending
    #
    # Tertiary:
    #   absolute forecast descending
    #
    # Final:
    #   coin alphabetically
    #
    # Dengan demikian, hasil tidak berubah hanya karena
    # urutan response API/thread berbeda.
    #

    results = sorted(
        results,
        key=lambda x: (
            -round(x["score"], 6),
            -round(x["confidence"], 6),
            -round(abs(x["forecast"]), 6),
            x["coin"],
        )
    )

    elapsed = time.time() - start

    return results, errors, elapsed


# ============================================================
# FORMAT HELPERS
# ============================================================

def fmt_price(x):

    if x >= 1000:
        return f"{x:,.2f}"

    if x >= 1:
        return f"{x:,.4f}"

    if x >= 0.01:
        return f"{x:.5f}"

    return f"{x:.8f}"


def signal_badge(signal):

    if signal == "LONG":
        return (
            '<span class="signal-long">🟢 LONG</span>'
        )

    return (
        '<span class="signal-short">🔴 SHORT</span>'
    )


def confidence_badge(label, value):

    if value >= 80:
        cls = "conf-strong"
    elif value >= 60:
        cls = "conf-medium"
    else:
        cls = "conf-weak"

    return (
        f'<span class="{cls}">'
        f'{label} · {value:.0f}%'
        f'</span>'
    )


def format_timestamp(ts):

    dt = datetime.fromtimestamp(
        ts / 1000,
        tz=timezone.utc
    )

    return dt.strftime(
        "%Y-%m-%d %H:%M UTC"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Scanner Settings")

    timeframe = st.selectbox(
        "Timeframe",
        list(TIMEFRAME_MAP.keys()),
        index=2
    )

    horizon_label = st.selectbox(
        "Horizon GPR",
        list(HORIZON_MAP.keys()),
        index=1
    )

    coin_count = st.selectbox(
        "Jumlah coin",
        [20, 40, 60, 79],
        index=3
    )

    minimum_score = st.slider(
        "Minimum GPR Score",
        min_value=40,
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
        value=5
    )

    st.divider()

    scan_button = st.button(
        "🚀 SCAN FUTURES",
        type="primary",
        use_container_width=True
    )

    clear_cache = st.button(
        "🗑️ CLEAR CACHE",
        use_container_width=True
    )

    if clear_cache:
        st.cache_data.clear()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("📈 GPR CRYPTO FUTURES SCANNER")

st.caption(
    f"Target utama: USDT Perpetual · "
    f"Timeframe {timeframe} · "
    f"GPR sebagai mesin ranking utama"
)


# ============================================================
# LOAD INSTRUMENTS
# ============================================================

try:

    all_instruments = get_swap_instruments()

except Exception as exc:

    st.error(
        f"Gagal mengambil daftar futures OKX: {exc}"
    )

    st.stop()


# ============================================================
# SELECT UNIVERSE
# ============================================================

available_map = {
    x["base"]: x
    for x in all_instruments
}

selected_instruments = []

for coin in DEFAULT_COINS:

    if coin in available_map:
        selected_instruments.append(
            available_map[coin]
        )

# Jika jumlah lebih banyak daripada static universe,
# tambahkan instrument lain secara alphabetical.

if len(selected_instruments) < coin_count:

    used = {
        x["base"]
        for x in selected_instruments
    }

    extras = [
        x
        for x in all_instruments
        if x["base"] not in used
    ]

    extras = sorted(
        extras,
        key=lambda x: x["base"]
    )

    selected_instruments.extend(
        extras[
            :coin_count - len(selected_instruments)
        ]
    )

selected_instruments = selected_instruments[
    :coin_count
]


# ============================================================
# TOP INFO
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "UNIVERSE",
        len(selected_instruments)
    )

with col2:
    st.metric(
        "TIMEFRAME",
        timeframe
    )

with col3:
    st.metric(
        "HORIZON",
        f"{HORIZON_MAP[horizon_label]} candle"
    )

with col4:
    st.metric(
        "PROVIDER",
        "OKX"
    )

st.info(
    f"Mode data: public market data, tanpa API key. "
    f"Provider OKX · {timeframe} · "
    f"Horizon {HORIZON_MAP[horizon_label]} candle."
)


# ============================================================
# SESSION STATE
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = []

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "scan_elapsed" not in st.session_state:
    st.session_state.scan_elapsed = 0


# ============================================================
# SCAN
# ============================================================

if scan_button:

    bar = TIMEFRAME_MAP[timeframe]
    horizon = HORIZON_MAP[horizon_label]

    with st.spinner(
        f"Scanning {len(selected_instruments)} futures..."
    ):

        results, errors, elapsed = scan_market(
            selected_instruments,
            bar,
            horizon,
            workers
        )

    st.session_state.scan_results = results
    st.session_state.scan_errors = errors
    st.session_state.scan_elapsed = elapsed


# ============================================================
# DISPLAY RESULTS
# ============================================================

results = st.session_state.scan_results
errors = st.session_state.scan_errors

if not results:

    st.warning(
        "Belum ada hasil scan. "
        "Tekan tombol 🚀 SCAN FUTURES."
    )

    st.stop()


# ============================================================
# FILTER
# ============================================================

qualified = [
    x
    for x in results
    if (
        x["score"] >= minimum_score
        and
        x["confidence"] >= minimum_confidence
    )
]


# ============================================================
# SCAN STATUS
# ============================================================

success_count = len(results)
error_count = len(errors)

st.success(
    f"Scan selesai: {success_count} berhasil, "
    f"{error_count} gagal dalam "
    f"{st.session_state.scan_elapsed:.1f} detik."
)


# ============================================================
# SUMMARY
# ============================================================

best = results[0]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "BERHASIL ANALISIS",
        success_count
    )

with c2:
    st.metric(
        f"LOLOS SCORE ≥ {minimum_score}",
        len(qualified)
    )

with c3:
    st.metric(
        "KANDIDAT",
        len(qualified)
    )

with c4:
    st.metric(
        "GPR RETURN TERTINGGI",
        f"{best['forecast']:+.2f}%",
        help="Forecast model, bukan jaminan profit."
    )


# ============================================================
# RANKING TABLE
# ============================================================

st.divider()

st.subheader("🏆 GPR Futures Ranking")

ranking_rows = []

for rank, item in enumerate(
    results,
    start=1
):

    ranking_rows.append({
        "Rank": rank,
        "Coin": item["coin"],
        "Signal": item["signal"],
        "GPR Score": round(item["score"], 1),
        "GPR Return": f"{item['forecast']:+.2f}%",
        "Uncertainty": f"{item['uncertainty']:.2f}%",
        "Confidence": f"{item['confidence']:.0f}%",
        "Price": fmt_price(item["price"]),
        "RSI": round(item["rsi"], 1),
        "Mom 6H": f"{item['mom6']:+.2f}%",
        "Mom 12H": f"{item['mom12']:+.2f}%",
        "Mom 24H": f"{item['mom24']:+.2f}%",
        "Vol Z": round(item["vol_z"], 2),
    })

ranking_df = pd.DataFrame(ranking_rows)

st.dataframe(
    ranking_df,
    use_container_width=True,
    hide_index=True,
    height=520
)


# ============================================================
# TOP SIGNALS
# ============================================================

st.divider()

st.subheader("🎯 Top GPR Signals")

top_results = qualified[:6]

if not top_results:

    st.warning(
        "Belum ada coin yang memenuhi kombinasi "
        f"Score ≥ {minimum_score} dan "
        f"Confidence ≥ {minimum_confidence}%."
    )

else:

    cards = st.columns(2)

    for i, item in enumerate(top_results):

        with cards[i % 2]:

            st.markdown(
                f"### {item['coin']}"
            )

            st.markdown(
                signal_badge(item["signal"]),
                unsafe_allow_html=True
            )

            st.markdown("")

            m1, m2 = st.columns(2)

            with m1:
                st.metric(
                    "GPR Score",
                    f"{item['score']:.1f}"
                )

            with m2:
                st.metric(
                    "Confidence",
                    f"{item['confidence']:.0f}%"
                )

            m3, m4 = st.columns(2)

            with m3:
                st.metric(
                    "GPR Return",
                    f"{item['forecast']:+.2f}%"
                )

            with m4:
                st.metric(
                    "Uncertainty",
                    f"{item['uncertainty']:.2f}%"
                )

            st.markdown(
                confidence_badge(
                    item["confidence_label"],
                    item["confidence"]
                ),
                unsafe_allow_html=True
            )

            st.caption(
                f"Signal agreement: "
                f"{item['agreement']:.0f}% · "
                f"Historical validation: "
                f"{item['historical_validation']:.0f}%"
            )


# ============================================================
# DETAIL ANALYSIS
# ============================================================

st.divider()

st.subheader("🔎 Detail Analysis")

coin_names = [
    x["coin"]
    for x in results
]

selected_coin = st.selectbox(
    "Pilih coin",
    coin_names
)

detail = next(
    x for x in results
    if x["coin"] == selected_coin
)


# ============================================================
# SIGNAL HEADER
# ============================================================

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric(
        "SIGNAL",
        detail["signal"]
    )

with d2:
    st.metric(
        "GPR SCORE",
        f"{detail['score']:.1f}"
    )

with d3:
    st.metric(
        "FORECAST",
        f"{detail['forecast']:+.2f}%"
    )

with d4:
    st.metric(
        "CONFIDENCE",
        f"{detail['confidence']:.0f}%"
    )


# ============================================================
# CONFIDENCE EXPLANATION
# ============================================================

st.markdown("### 📊 Cara Membaca Confidence")

confidence = detail["confidence"]

if confidence >= 80:

    st.success(
        f"SANGAT KUAT — Confidence {confidence:.0f}%. "
        "Komponen model relatif konsisten mendukung arah signal."
    )

elif confidence >= 65:

    st.info(
        f"KUAT — Confidence {confidence:.0f}%. "
        "Dukungan model cukup baik, tetapi belum masuk kategori sangat kuat."
    )

elif confidence >= 50:

    st.warning(
        f"SEDANG — Confidence {confidence:.0f}%. "
        "Ada dukungan terhadap signal, tetapi konflik antar komponen masih cukup besar."
    )

else:

    st.error(
        f"LEMAH — Confidence {confidence:.0f}%. "
        "Signal sebaiknya dianggap kurang meyakinkan."
    )


st.caption(
    "Confidence bukan probabilitas pasti profit. "
    "Angka ini adalah ukuran kekuatan dukungan model "
    "berdasarkan komponen teknikal dan validasi historis."
)


# ============================================================
# TECHNICAL DATA
# ============================================================

st.markdown("### 📐 Technical Data")

t1, t2, t3, t4 = st.columns(4)

with t1:
    st.metric(
        "Harga",
        fmt_price(detail["price"])
    )

with t2:
    st.metric(
        "RSI 14",
        f"{detail['rsi']:.1f}"
    )

with t3:
    st.metric(
        "Momentum 6H",
        f"{detail['mom6']:+.2f}%"
    )

with t4:
    st.metric(
        "Momentum 12H",
        f"{detail['mom12']:+.2f}%"
    )

t5, t6, t7, t8 = st.columns(4)

with t5:
    st.metric(
        "Momentum 24H",
        f"{detail['mom24']:+.2f}%"
    )

with t6:
    st.metric(
        "Volume Z",
        f"{detail['vol_z']:+.2f}"
    )

with t7:
    st.metric(
        "EMA12 Distance",
        f"{detail['ema12_dist']:+.2f}%"
    )

with t8:
    st.metric(
        "EMA24 Distance",
        f"{detail['ema24_dist']:+.2f}%"
    )

st.caption(
    "Last confirmed candle: "
    + format_timestamp(detail["timestamp"])
)


# ============================================================
# CONFIDENCE COMPONENTS
# ============================================================

st.divider()

st.markdown("### 🧪 GPR Signal Components")

component_df = pd.DataFrame({
    "Component": [
        "Momentum",
        "Trend EMA",
        "RSI",
        "Volume",
        "Signal Agreement",
        "Historical Validation",
        "Trend Consistency",
    ],

    "Value": [
        round(detail["momentum_component"], 4),
        round(detail["trend_component"], 4),
        round(detail["rsi_component"], 4),
        round(detail["volume_component"], 4),
        f"{detail['agreement']:.0f}%",
        f"{detail['historical_validation']:.0f}%",
        "YES" if detail["trend_consistency"] else "NO",
    ]
})

st.dataframe(
    component_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RISK MANAGEMENT
# ============================================================

st.divider()

st.markdown("### 🎯 Risk Management")

risk_reward = st.slider(
    "Risk / Reward Target",
    min_value=1.0,
    max_value=4.0,
    value=1.67,
    step=0.01
)

entry = detail["price"]

# Risk distance berdasarkan uncertainty.
risk_pct = max(
    detail["uncertainty"] / 100,
    0.003
)

if detail["signal"] == "LONG":

    stop_loss = entry * (
        1 - risk_pct
    )

    take_profit = entry * (
        1 + risk_pct * risk_reward
    )

else:

    stop_loss = entry * (
        1 + risk_pct
    )

    take_profit = entry * (
        1 - risk_pct * risk_reward
    )


r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric(
        "Entry / Current",
        fmt_price(entry)
    )

with r2:
    st.metric(
        "Stop Loss",
        fmt_price(stop_loss)
    )

with r3:
    st.metric(
        "Take Profit",
        fmt_price(take_profit)
    )

with r4:
    st.metric(
        "Risk / Reward",
        f"1 : {risk_reward:.2f}"
    )

st.caption(
    "Stop Loss dihitung dari model uncertainty. "
    "Take Profit mengikuti target Risk/Reward. "
    "Perhitungan ini adalah alat bantu analisis, bukan instruksi trading otomatis."
)


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.markdown("### 📝 Interpretasi")

direction_word = (
    "bullish"
    if detail["signal"] == "LONG"
    else "bearish"
)

agreement_text = (
    "searah"
    if detail["agreement"] >= 75
    else "belum sepenuhnya searah"
)

st.write(
    f"**{detail['coin']}** memiliki GPR Score "
    f"**{detail['score']:.1f}** dengan Confidence "
    f"**{detail['confidence']:.0f}%**. "
    f"Bias utama saat ini adalah **{detail['signal'].lower()}**."
)

st.write(
    f"Momentum dan trend memberikan kondisi "
    f"**{direction_word}**, sementara tingkat agreement "
    f"antar komponen berada di **{detail['agreement']:.0f}%**."
)

st.write(
    f"Validasi historis model pada kondisi serupa "
    f"berada di sekitar **{detail['historical_validation']:.0f}%**."
)

if (
    detail["agreement"] >= 75
    and detail["historical_validation"] >= 55
):

    st.success(
        "Kesimpulan model: signal relatif konsisten "
        "dan didukung oleh beberapa komponen."
    )

elif detail["agreement"] >= 50:

    st.warning(
        "Kesimpulan model: signal ada, tetapi "
        "dukungan antar komponen belum cukup kuat."
    )

else:

    st.error(
        "Kesimpulan model: signal lemah dan "
        "komponen indikator banyak berbeda arah."
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

st.divider()

with st.expander(
    f"🧾 ERROR / DIAGNOSTIC ({len(errors)})"
):

    if not errors:

        st.success(
            "Tidak ada error pada coin yang dianalisis."
        )

    else:

        diagnostic_rows = []

        for err in errors:

            diagnostic_rows.append({
                "Symbol": err.get("coin"),
                "Instrument": err.get("inst_id"),
                "Status": err.get("error"),
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
    "GPR Crypto Futures Scanner · "
    "USDT Perpetual · "
    f"{timeframe} · "
    "OKX Public Market Data · "
    "Research Prototype"
)
