import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ============================================================
# GPR CRYPTO FUTURES SCANNER
# OKX PUBLIC MARKET DATA
# TIMEFRAME: 15m
# HORIZON: 10 CANDLES
# ============================================================

st.set_page_config(
    page_title="GPR Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONFIG
# ============================================================

OKX_BASE_URL = "https://openapi.okx.com"

DEFAULT_TIMEFRAME = "15m"
DEFAULT_HORIZON = 10
DEFAULT_COINS = 79
DEFAULT_MIN_SCORE = 70
DEFAULT_MIN_CONFIDENCE = 60
DEFAULT_WORKERS = 5

CANDLE_LIMIT = 150
REQUEST_TIMEOUT = 15

# ============================================================
# SESSION STATE
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "scan_timestamp" not in st.session_state:
    st.session_state.scan_timestamp = None

if "selected_coin" not in st.session_state:
    st.session_state.selected_coin = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .info-box {
        padding: 14px 18px;
        border-radius: 8px;
        background: #e8f2ff;
        border: 1px solid #c9e0ff;
        color: #0757a6;
        margin-bottom: 15px;
    }

    .success-box {
        padding: 14px 18px;
        border-radius: 8px;
        background: #eaf8ef;
        border: 1px solid #c8ead5;
        color: #087a3d;
        margin-bottom: 15px;
    }

    .warning-box {
        padding: 14px 18px;
        border-radius: 8px;
        background: #fff8df;
        border: 1px solid #f2df9a;
        color: #806000;
        margin-bottom: 15px;
    }

    .metric-card {
        padding: 18px;
        border: 1px solid #d9dee7;
        border-radius: 10px;
        background: white;
        min-height: 115px;
    }

    .metric-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 700;
        margin-top: 6px;
    }

    .metric-sub {
        font-size: 12px;
        color: #8a919e;
        margin-top: 5px;
    }

    .long-text {
        color: #087f3f;
        font-weight: 700;
    }

    .short-text {
        color: #c62828;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .signal-card {
        border: 1px solid #d9dee7;
        border-radius: 12px;
        padding: 20px;
        background: white;
        margin-bottom: 15px;
    }

    .signal-coin {
        font-size: 24px;
        font-weight: 800;
    }

    .small-muted {
        color: #8a919e;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📈 GPR CRYPTO FUTURES SCANNER</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Target utama: USDT Perpetual · Timeframe 15m · "
    "GPR sebagai mesin ranking utama"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Scanner Settings")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["5m", "15m", "30m", "1H", "2H", "4H"],
    index=1,
)

horizon_label = st.sidebar.selectbox(
    "Horizon GPR",
    [
        "5 candle",
        "10 candle / 2.5 jam",
        "15 candle / 3.75 jam",
        "20 candle / 5 jam",
        "30 candle / 7.5 jam",
    ],
    index=1,
)

horizon_map = {
    "5 candle": 5,
    "10 candle / 2.5 jam": 10,
    "15 candle / 3.75 jam": 15,
    "20 candle / 5 jam": 20,
    "30 candle / 7.5 jam": 30,
}

horizon = horizon_map[horizon_label]

coin_count = st.sidebar.selectbox(
    "Jumlah coin",
    [20, 30, 50, 79, 100],
    index=3,
)

min_score = st.sidebar.slider(
    "Minimum GPR Score",
    min_value=50,
    max_value=95,
    value=DEFAULT_MIN_SCORE,
    step=1,
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=40,
    max_value=95,
    value=DEFAULT_MIN_CONFIDENCE,
    step=1,
)

workers = st.sidebar.slider(
    "Parallel workers",
    min_value=1,
    max_value=10,
    value=DEFAULT_WORKERS,
    step=1,
)

st.sidebar.divider()

scan_button = st.sidebar.button(
    "🚀 SCAN FUTURES",
    type="primary",
    use_container_width=True,
)

clear_button = st.sidebar.button(
    "🗑️ CLEAR CACHE",
    use_container_width=True,
)


# ============================================================
# CLEAR
# ============================================================

if clear_button:
    st.session_state.scan_results = None
    st.session_state.scan_errors = []
    st.session_state.scan_timestamp = None
    st.session_state.selected_coin = None

    try:
        st.cache_data.clear()
    except Exception:
        pass

    st.rerun()


# ============================================================
# UTILITY
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def timeframe_minutes(tf):
    mapping = {
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1H": 60,
        "2H": 120,
        "4H": 240,
    }
    return mapping.get(tf, 15)


def format_price(value):
    if pd.isna(value):
        return "-"
    value = float(value)

    if value >= 1000:
        return f"{value:,.2f}"
    elif value >= 1:
        return f"{value:,.4f}"
    elif value >= 0.01:
        return f"{value:.5f}"
    else:
        return f"{value:.8f}"


def format_pct(value):
    if pd.isna(value):
        return "-"
    return f"{value:+.2f}%"


# ============================================================
# OKX PUBLIC API
# ============================================================

@st.cache_data(ttl=120, show_spinner=False)
def get_okx_instruments():
    """
    Get active USDT perpetual instruments.
    """

    url = f"{OKX_BASE_URL}/api/v5/public/instruments"

    params = {
        "instType": "SWAP"
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "GPR-Crypto-Futures-Scanner/1.0"
        },
    )

    response.raise_for_status()

    payload = response.json()

    if str(payload.get("code")) != "0":
        raise RuntimeError(
            f"OKX instruments error: "
            f"{payload.get('code')} {payload.get('msg')}"
        )

    rows = []

    for item in payload.get("data", []):
        inst_id = item.get("instId", "")
        state = item.get("state", "")
        settle_ccy = item.get("settleCcy", "")

        if (
            inst_id.endswith("-USDT-SWAP")
            and settle_ccy == "USDT"
            and state == "live"
        ):
            rows.append(
                {
                    "instId": inst_id,
                    "symbol": inst_id.replace("-USDT-SWAP", ""),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(ttl=30, show_spinner=False)
def get_tickers():
    """
    Get all live SWAP tickers.
    """

    url = f"{OKX_BASE_URL}/api/v5/market/tickers"

    params = {
        "instType": "SWAP"
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "GPR-Crypto-Futures-Scanner/1.0"
        },
    )

    response.raise_for_status()

    payload = response.json()

    if str(payload.get("code")) != "0":
        raise RuntimeError(
            f"OKX ticker error: "
            f"{payload.get('code')} {payload.get('msg')}"
        )

    rows = []

    for item in payload.get("data", []):
        inst_id = item.get("instId", "")

        if not inst_id.endswith("-USDT-SWAP"):
            continue

        rows.append(
            {
                "instId": inst_id,
                "last": safe_float(item.get("last")),
                "vol24h": safe_float(item.get("vol24h")),
                "volCcy24h": safe_float(item.get("volCcy24h")),
            }
        )

    return pd.DataFrame(rows)


def get_candles(inst_id, bar="15m", limit=150):
    """
    Get recent candles.

    OKX format:
    [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
    """

    url = f"{OKX_BASE_URL}/api/v5/market/candles"

    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": str(min(limit, 300)),
    }

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "GPR-Crypto-Futures-Scanner/1.0"
        },
    )

    response.raise_for_status()

    payload = response.json()

    if str(payload.get("code")) != "0":
        raise RuntimeError(
            f"OKX candles error: "
            f"{payload.get('code')} {payload.get('msg')}"
        )

    data = payload.get("data", [])

    if not data:
        raise RuntimeError("Tidak ada candle.")

    columns = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol_ccy",
        "vol_ccy_quote",
        "confirm",
    ]

    df = pd.DataFrame(data, columns=columns)

    for col in columns[1:]:
        if col != "confirm":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    df["ts"] = pd.to_numeric(
        df["ts"],
        errors="coerce",
    )

    df["datetime"] = pd.to_datetime(
        df["ts"],
        unit="ms",
        utc=True,
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    # Hanya gunakan candle yang sudah confirmed
    if "confirm" in df.columns:
        confirmed = df["confirm"].astype(str) == "1"

        if confirmed.sum() >= 30:
            df = df.loc[confirmed].copy()

    df = df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    ).reset_index(drop=True)

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_zscore(series, window=30):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    z = (series - mean) / std.replace(0, np.nan)

    return z


def linear_forecast(close, horizon):
    """
    Lightweight regression forecast.

    Menggunakan log-price agar forecast relatif
    lebih stabil terhadap perbedaan harga antar coin.
    """

    y = np.log(close.values.astype(float))

    n = len(y)

    lookback = min(60, n)

    y_fit = y[-lookback:]

    x = np.arange(lookback, dtype=float)

    if lookback < 10:
        return np.nan, np.nan

    try:
        slope, intercept = np.polyfit(x, y_fit, 1)
    except Exception:
        return np.nan, np.nan

    future_x = lookback + horizon - 1

    predicted_log = intercept + slope * future_x

    current_log = y[-1]

    forecast_return = (
        math.exp(predicted_log - current_log) - 1
    ) * 100

    fitted = intercept + slope * x

    residuals = y_fit - fitted

    residual_std = np.std(residuals)

    uncertainty = (
        residual_std * math.sqrt(max(horizon, 1)) * 100
    )

    return forecast_return, uncertainty


# ============================================================
# FEATURE ENGINE
# ============================================================

def build_features(df, horizon):
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # EMA
    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema24 = close.ewm(
        span=24,
        adjust=False,
    ).mean()

    # RSI
    rsi = calculate_rsi(close, 14)

    # Momentum
    bars_6h = max(
        1,
        int((6 * 60) / timeframe_minutes(current_timeframe))
    )

    bars_12h = max(
        1,
        int((12 * 60) / timeframe_minutes(current_timeframe))
    )

    bars_24h = max(
        1,
        int((24 * 60) / timeframe_minutes(current_timeframe))
    )

    mom6 = (
        close.iloc[-1] /
        close.iloc[-1 - bars_6h] - 1
    ) * 100 if len(close) > bars_6h else np.nan

    mom12 = (
        close.iloc[-1] /
        close.iloc[-1 - bars_12h] - 1
    ) * 100 if len(close) > bars_12h else np.nan

    mom24 = (
        close.iloc[-1] /
        close.iloc[-1 - bars_24h] - 1
    ) * 100 if len(close) > bars_24h else np.nan

    # Volume Z
    volume_z = calculate_zscore(
        volume,
        30,
    ).iloc[-1]

    # EMA distance
    ema12_distance = (
        (close.iloc[-1] / ema12.iloc[-1]) - 1
    ) * 100

    ema24_distance = (
        (close.iloc[-1] / ema24.iloc[-1]) - 1
    ) * 100

    # Forecast
    forecast, model_uncertainty = linear_forecast(
        close,
        horizon,
    )

    # Historical volatility
    returns = close.pct_change().dropna()

    if len(returns) >= 30:
        volatility = (
            returns.tail(30).std()
            * 100
            * math.sqrt(horizon)
        )
    else:
        volatility = np.nan

    # Combine uncertainty
    uncertainty = np.nanmean(
        [
            abs(model_uncertainty)
            if not pd.isna(model_uncertainty)
            else np.nan,
            abs(volatility)
            if not pd.isna(volatility)
            else np.nan,
        ]
    )

    if pd.isna(uncertainty):
        uncertainty = 1.0

    # Avoid zero
    uncertainty = max(float(uncertainty), 0.05)

    return {
        "price": close.iloc[-1],
        "rsi": rsi.iloc[-1],
        "mom6": mom6,
        "mom12": mom12,
        "mom24": mom24,
        "volume_z": volume_z,
        "ema12_distance": ema12_distance,
        "ema24_distance": ema24_distance,
        "forecast": forecast,
        "uncertainty": uncertainty,
        "last_candle": df["datetime"].iloc[-1],
    }


# ============================================================
# GPR SCORING
# ============================================================

def calculate_gpr(features):
    """
    GPR-style directional scoring.

    Score:
        0 - 100

    Confidence:
        0 - 100

    Forecast:
        predicted directional return

    Important:
    Confidence bukan probabilitas profit.
    """

    forecast = safe_float(features["forecast"], 0.0)
    mom6 = safe_float(features["mom6"], 0.0)
    mom12 = safe_float(features["mom12"], 0.0)
    mom24 = safe_float(features["mom24"], 0.0)

    rsi = safe_float(features["rsi"], 50.0)

    volume_z = safe_float(
        features["volume_z"],
        0.0,
    )

    ema12 = safe_float(
        features["ema12_distance"],
        0.0,
    )

    ema24 = safe_float(
        features["ema24_distance"],
        0.0,
    )

    uncertainty = max(
        safe_float(features["uncertainty"], 1.0),
        0.05,
    )

    # --------------------------------------------------------
    # DIRECTIONAL COMPONENTS
    # --------------------------------------------------------

    forecast_component = np.tanh(
        forecast / max(uncertainty * 1.5, 0.5)
    )

    momentum_component = np.tanh(
        (
            0.30 * mom6
            + 0.35 * mom12
            + 0.35 * mom24
        ) / 3.0
    )

    ema_component = np.tanh(
        (
            0.45 * ema12
            + 0.55 * ema24
        ) / 1.5
    )

    # RSI:
    # >50 mendukung bullish
    # <50 mendukung bearish
    rsi_component = np.tanh(
        (rsi - 50) / 12
    )

    volume_component = np.tanh(
        volume_z / 2
    )

    # --------------------------------------------------------
    # COMBINED DIRECTION
    # --------------------------------------------------------

    directional_signal = (
        0.35 * forecast_component
        + 0.25 * momentum_component
        + 0.20 * ema_component
        + 0.10 * rsi_component
        + 0.10 * volume_component
    )

    directional_signal = clamp(
        directional_signal,
        -1,
        1,
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    signal = (
        "LONG"
        if directional_signal >= 0
        else "SHORT"
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 50 + abs(directional_signal) * 50

    # Forecast magnitude bonus
    forecast_strength = min(
        abs(forecast) / max(uncertainty, 0.1),
        3.0,
    )

    score += min(
        forecast_strength * 4,
        12,
    )

    score = clamp(
        score,
        0,
        100,
    )

    # --------------------------------------------------------
    # SIGNAL AGREEMENT
    # --------------------------------------------------------

    components = [
        np.sign(forecast),
        np.sign(mom6),
        np.sign(mom12),
        np.sign(mom24),
        np.sign(ema12),
        np.sign(ema24),
        np.sign(rsi - 50),
    ]

    target_direction = (
        1
        if signal == "LONG"
        else -1
    )

    valid_components = [
        x
        for x in components
        if x != 0
    ]

    if valid_components:
        agreement = (
            sum(
                1
                for x in valid_components
                if x == target_direction
            )
            / len(valid_components)
        ) * 100
    else:
        agreement = 50.0

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    forecast_conf = clamp(
        abs(forecast) /
        max(uncertainty * 2, 0.5),
        0,
        1,
    ) * 100

    uncertainty_quality = clamp(
        1 - uncertainty / 5,
        0,
        1,
    ) * 100

    confidence = (
        0.45 * agreement
        + 0.35 * forecast_conf
        + 0.20 * uncertainty_quality
    )

    confidence = clamp(
        confidence,
        0,
        100,
    )

    # --------------------------------------------------------
    # CONFIDENCE LABEL
    # --------------------------------------------------------

    if confidence >= 85:
        confidence_label = "SANGAT KUAT"
    elif confidence >= 75:
        confidence_label = "KUAT"
    elif confidence >= 65:
        confidence_label = "BAIK"
    elif confidence >= 55:
        confidence_label = "SEDANG"
    else:
        confidence_label = "LEMAH"

    return {
        "signal": signal,
        "score": round(score, 1),
        "confidence": round(confidence, 1),
        "confidence_label": confidence_label,
        "agreement": round(agreement, 1),
        "forecast": round(forecast, 2),
        "uncertainty": round(uncertainty, 2),
    }


# ============================================================
# SINGLE COIN SCAN
# ============================================================

def scan_coin(inst_id, symbol, bar, horizon):
    try:
        df = get_candles(
            inst_id,
            bar=bar,
            limit=CANDLE_LIMIT,
        )

        if len(df) < 60:
            raise RuntimeError(
                f"Candle tidak cukup: {len(df)}"
            )

        features = build_features(
            df,
            horizon,
        )

        gpr = calculate_gpr(
            features
        )

        result = {
            "COIN": symbol,
            "INST_ID": inst_id,
            "SIGNAL": gpr["signal"],
            "GPR SCORE": gpr["score"],
            "GPR RETURN": gpr["forecast"],
            "UNCERTAINTY": gpr["uncertainty"],
            "CONFIDENCE": gpr["confidence"],
            "CONF LABEL": gpr["confidence_label"],
            "SIGNAL AGREEMENT": gpr["agreement"],
            "PRICE": features["price"],
            "RSI": features["rsi"],
            "MOM 6H": features["mom6"],
            "MOM 12H": features["mom12"],
            "MOM 24H": features["mom24"],
            "VOL Z": features["volume_z"],
            "EMA12 DIST": features["ema12_distance"],
            "EMA24 DIST": features["ema24_distance"],
            "LAST CANDLE": features["last_candle"],
        }

        return result, None

    except Exception as e:
        return None, {
            "symbol": symbol,
            "status": str(e),
        }


# ============================================================
# SCAN ENGINE
# ============================================================

def run_scan(bar, horizon, coin_count, workers):
    """
    Scan selected USDT perpetual coins.

    Penting:
    - Instruments diambil dari OKX
    - Ticker dipakai untuk ranking volume
    - Coin dipilih berdasarkan volume 24h
    - Candle diproses paralel
    """

    instruments = get_okx_instruments()

    if instruments.empty:
        raise RuntimeError(
            "OKX tidak mengembalikan USDT perpetual."
        )

    tickers = get_tickers()

    if tickers.empty:
        raise RuntimeError(
            "OKX tidak mengembalikan ticker."
        )

    universe = instruments.merge(
        tickers,
        on="instId",
        how="left",
    )

    universe["volCcy24h"] = universe[
        "volCcy24h"
    ].fillna(0)

    universe = universe.sort_values(
        "volCcy24h",
        ascending=False,
    )

    universe = universe.head(
        min(coin_count, len(universe))
    ).copy()

    results = []
    errors = []

    total = len(universe)

    progress = st.progress(
        0,
        text="Memulai scan...",
    )

    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {}

        for _, row in universe.iterrows():

            future = executor.submit(
                scan_coin,
                row["instId"],
                row["symbol"],
                bar,
                horizon,
            )

            futures[future] = row["symbol"]

        completed = 0

        for future in as_completed(futures):

            completed += 1

            symbol = futures[future]

            try:
                result, error = future.result()

                if result is not None:
                    results.append(result)

                if error is not None:
                    errors.append(error)

            except Exception as e:
                errors.append(
                    {
                        "symbol": symbol,
                        "status": str(e),
                    }
                )

            progress.progress(
                completed / max(total, 1),
                text=(
                    f"Scanning {completed}/{total} "
                    f"· {symbol}"
                ),
            )

            status.write(
                f"Berhasil: {len(results)} · "
                f"Gagal: {len(errors)}"
            )

    progress.empty()
    status.empty()

    if not results:
        raise RuntimeError(
            "Semua coin gagal dianalisis. "
            "Lihat bagian ERROR / DIAGNOSTIC."
        )

    result_df = pd.DataFrame(results)

    # ========================================================
    # DETERMINISTIC RANKING
    # ========================================================

    result_df = result_df.sort_values(
        by=[
            "GPR SCORE",
            "CONFIDENCE",
            "SIGNAL AGREEMENT",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    result_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(result_df) + 1,
        ),
    )

    return result_df, errors


# ============================================================
# SCAN ACTION
# ============================================================

current_timeframe = timeframe

if scan_button:

    # Hapus hasil lama agar tidak tercampur
    st.session_state.scan_results = None
    st.session_state.scan_errors = []
    st.session_state.scan_timestamp = None
    st.session_state.selected_coin = None

    try:

        with st.spinner(
            "Mengambil market data OKX dan menjalankan GPR..."
        ):

            results, errors = run_scan(
                timeframe,
                horizon,
                coin_count,
                workers,
            )

        st.session_state.scan_results = results
        st.session_state.scan_errors = errors
        st.session_state.scan_timestamp = datetime.now(
            timezone.utc
        )

        if not results.empty:
            st.session_state.selected_coin = (
                results.iloc[0]["COIN"]
            )

        st.rerun()

    except Exception as e:

        st.session_state.scan_results = None

        st.error(
            f"❌ Scan gagal: {e}"
        )


# ============================================================
# MAIN DISPLAY
# ============================================================

results = st.session_state.scan_results
errors = st.session_state.scan_errors


# ============================================================
# TOP INFO
# ============================================================

st.markdown(
    '<div class="info-box">'
    f"Mode data: <b>public market data</b>, tanpa API key. "
    f"Provider <b>OKX</b> · "
    f"Timeframe <b>{timeframe}</b> · "
    f"Horizon <b>{horizon} candle</b>."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# NO RESULT
# ============================================================

if results is None:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "UNIVERSE",
            "0",
            "USDT Perpetual",
        )

    with col2:
        st.metric(
            "TIMEFRAME",
            timeframe,
            "Primary timeframe",
        )

    with col3:
        st.metric(
            "HORIZON",
            f"{horizon} candle",
            f"≈ {horizon * timeframe_minutes(timeframe) / 60:.2f} jam",
        )

    with col4:
        st.metric(
            "PROVIDER",
            "OKX",
            "Public market data",
        )

    st.markdown(
        '<div class="warning-box">'
        "Belum ada hasil scan. "
        "Tekan tombol 🚀 <b>SCAN FUTURES</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# SUMMARY
# ============================================================

success_count = len(results)
error_count = len(errors)

qualified = results[
    (results["GPR SCORE"] >= min_score)
    & (results["CONFIDENCE"] >= min_confidence)
].copy()

candidate_count = len(qualified)

if success_count > 0:
    best = results.iloc[0]

    highest_return_row = results.loc[
        results["GPR RETURN"].idxmax()
    ]

    best_return = highest_return_row[
        "GPR RETURN"
    ]

else:
    best = None
    best_return = np.nan


st.markdown(
    '<div class="success-box">'
    f"Scan selesai: <b>{success_count}</b> berhasil, "
    f"<b>{error_count}</b> gagal. "
    f"Minimum Score {min_score} · "
    f"Minimum Confidence {min_confidence}%."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# KPI ROW 1
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "UNIVERSE",
        success_count,
        "USDT Perpetual",
    )

with c2:
    st.metric(
        "TIMEFRAME",
        timeframe,
        "Primary timeframe",
    )

with c3:
    st.metric(
        "HORIZON",
        f"{horizon} candle",
        f"≈ {horizon * timeframe_minutes(timeframe) / 60:.2f} jam",
    )

with c4:
    st.metric(
        "PROVIDER",
        "OKX",
        "Public market data",
    )


# ============================================================
# KPI ROW 2
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "BERHASIL DIANALISIS",
        success_count,
    )

with c2:
    st.metric(
        f"LOLOS SCORE ≥ {min_score}",
        int(
            (results["GPR SCORE"] >= min_score).sum()
        ),
    )

with c3:
    st.metric(
        "KANDIDAT",
        candidate_count,
        f"Score ≥ {min_score} + Conf ≥ {min_confidence}%",
    )

with c4:
    st.metric(
        "GPR RETURN TERTINGGI",
        format_pct(best_return),
        (
            f"Score "
            f"{highest_return_row['GPR SCORE']:.1f}"
            if best is not None
            else ""
        ),
    )


# ============================================================
# RANKING
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🏆 GPR Futures Ranking</div>',
    unsafe_allow_html=True,
)

ranking = results.copy()

display_cols = [
    "Rank",
    "COIN",
    "SIGNAL",
    "GPR SCORE",
    "GPR RETURN",
    "UNCERTAINTY",
    "CONFIDENCE",
    "PRICE",
    "RSI",
    "MOM 6H",
    "MOM 12H",
    "MOM 24H",
    "VOL Z",
]

display_df = ranking[display_cols].copy()

display_df["GPR RETURN"] = display_df[
    "GPR RETURN"
].map(lambda x: f"{x:+.2f}%")

display_df["UNCERTAINTY"] = display_df[
    "UNCERTAINTY"
].map(lambda x: f"{x:.2f}%")

display_df["CONFIDENCE"] = display_df[
    "CONFIDENCE"
].map(lambda x: f"{x:.1f}%")

display_df["PRICE"] = display_df[
    "PRICE"
].map(format_price)

display_df["RSI"] = display_df[
    "RSI"
].map(lambda x: f"{x:.1f}")

for col in [
    "MOM 6H",
    "MOM 12H",
    "MOM 24H",
]:
    display_df[col] = display_df[
        col
    ].map(lambda x: f"{x:+.2f}%")

display_df["VOL Z"] = display_df[
    "VOL Z"
].map(lambda x: f"{x:+.2f}")

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=520,
)


# ============================================================
# TOP SIGNALS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🎯 Top GPR Signals</div>',
    unsafe_allow_html=True,
)

top_signals = qualified.head(6)

if top_signals.empty:

    st.warning(
        "Belum ada coin yang memenuhi kombinasi "
        f"GPR Score ≥ {min_score} dan "
        f"Confidence ≥ {min_confidence}%."
    )

else:

    columns = st.columns(2)

    for i, (_, row) in enumerate(
        top_signals.iterrows()
    ):

        col = columns[i % 2]

        signal_class = (
            "long-text"
            if row["SIGNAL"] == "LONG"
            else "short-text"
        )

        signal_icon = (
            "🟢"
            if row["SIGNAL"] == "LONG"
            else "🔴"
        )

        with col:

            st.markdown(
                f"""
                <div class="signal-card">

                <div class="signal-coin">
                    {row["COIN"]}
                </div>

                <div style="margin-top:8px;">
                    <span class="{signal_class}">
                    {signal_icon} {row["SIGNAL"]}
                    </span>
                </div>

                <hr>

                <b>GPR Score</b>
                <div style="font-size:30px;">
                    {row["GPR SCORE"]:.1f}
                </div>

                <div style="margin-top:8px;">
                    <b>GPR Return:</b>
                    {row["GPR RETURN"]:+.2f}%
                </div>

                <div style="margin-top:5px;">
                    <b>Uncertainty:</b>
                    {row["UNCERTAINTY"]:.2f}%
                </div>

                <div style="margin-top:5px;">
                    <b>Confidence:</b>
                    {row["CONFIDENCE"]:.1f}%
                    · {row["CONF LABEL"]}
                </div>

                <div style="margin-top:5px;">
                    <b>Signal Agreement:</b>
                    {row["SIGNAL AGREEMENT"]:.1f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DETAIL ANALYSIS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🔎 Detail Analysis</div>',
    unsafe_allow_html=True,
)

coin_list = results["COIN"].tolist()

default_index = 0

if (
    st.session_state.selected_coin
    and st.session_state.selected_coin in coin_list
):
    default_index = coin_list.index(
        st.session_state.selected_coin
    )

selected_coin = st.selectbox(
    "Pilih coin",
    coin_list,
    index=default_index,
)

st.session_state.selected_coin = selected_coin

selected = results[
    results["COIN"] == selected_coin
].iloc[0]


# ============================================================
# SIGNAL HEADER
# ============================================================

signal = selected["SIGNAL"]

signal_color = (
    "🟢 LONG"
    if signal == "LONG"
    else "🔴 SHORT"
)

a, b, c, d = st.columns(4)

with a:
    st.metric(
        "SIGNAL",
        signal_color,
    )

with b:
    st.metric(
        "GPR SCORE",
        f"{selected['GPR SCORE']:.1f}",
    )

with c:
    st.metric(
        "FORECAST",
        f"{selected['GPR RETURN']:+.2f}%",
    )

with d:
    st.metric(
        "UNCERTAINTY",
        f"{selected['UNCERTAINTY']:.2f}%",
    )


# ============================================================
# CONFIDENCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 GPR Confidence</div>',
    unsafe_allow_html=True,
)

confidence = float(
    selected["CONFIDENCE"]
)

confidence_label = selected[
    "CONF LABEL"
]

st.info(
    f"**{confidence_label} — Confidence {confidence:.1f}%**. "
    "Confidence menunjukkan seberapa kuat kombinasi "
    "indikator mendukung arah signal. "
    "Ini **bukan probabilitas pasti profit**."
)

cc1, cc2, cc3, cc4 = st.columns(4)

with cc1:
    st.metric(
        "Confidence",
        f"{confidence:.1f}%",
    )

with cc2:
    st.metric(
        "Signal Agreement",
        f"{selected['SIGNAL AGREEMENT']:.1f}%",
    )

with cc3:
    st.metric(
        "Uncertainty",
        f"{selected['UNCERTAINTY']:.2f}%",
    )

with cc4:
    ratio = (
        abs(selected["GPR RETURN"])
        / max(selected["UNCERTAINTY"], 0.01)
    )

    st.metric(
        "Forecast / Uncertainty",
        f"{ratio:.2f}x",
    )


st.progress(
    min(max(confidence / 100, 0), 1)
)


# ============================================================
# TECHNICAL DATA
# ============================================================

st.markdown(
    '<div class="section-title">📐 Technical Data</div>',
    unsafe_allow_html=True,
)

t1, t2, t3, t4 = st.columns(4)

with t1:
    st.metric(
        "Harga",
        format_price(selected["PRICE"]),
    )

with t2:
    st.metric(
        "RSI 14",
        f"{selected['RSI']:.1f}",
    )

with t3:
    st.metric(
        "Momentum 6H",
        f"{selected['MOM 6H']:+.2f}%",
    )

with t4:
    st.metric(
        "Momentum 12H",
        f"{selected['MOM 12H']:+.2f}%",
    )

t1, t2, t3, t4 = st.columns(4)

with t1:
    st.metric(
        "Momentum 24H",
        f"{selected['MOM 24H']:+.2f}%",
    )

with t2:
    st.metric(
        "Volume Z",
        f"{selected['VOL Z']:+.2f}",
    )

with t3:
    st.metric(
        "EMA12 Distance",
        f"{selected['EMA12 DIST']:+.2f}%",
    )

with t4:
    st.metric(
        "EMA24 Distance",
        f"{selected['EMA24 DIST']:+.2f}%",
    )

st.caption(
    "Last confirmed candle: "
    + str(selected["LAST CANDLE"])
)


# ============================================================
# RISK MANAGEMENT
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🎯 Risk Management</div>',
    unsafe_allow_html=True,
)

risk_reward = st.slider(
    "Risk / Reward Target",
    min_value=1.0,
    max_value=5.0,
    value=1.67,
    step=0.01,
)

entry = float(
    selected["PRICE"]
)

uncertainty = float(
    selected["UNCERTAINTY"]
)

signal = selected["SIGNAL"]

# ------------------------------------------------------------
# Stop distance
# ------------------------------------------------------------

# uncertainty adalah percentage.
# Kita tambahkan buffer agar SL tidak terlalu dekat.

stop_distance_pct = max(
    uncertainty * 1.5,
    0.30,
)

stop_distance = (
    entry * stop_distance_pct / 100
)

if signal == "LONG":

    stop_loss = entry - stop_distance

    take_profit = (
        entry
        + stop_distance * risk_reward
    )

else:

    stop_loss = entry + stop_distance

    take_profit = (
        entry
        - stop_distance * risk_reward
    )


r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric(
        "Entry / Current",
        format_price(entry),
    )

with r2:
    st.metric(
        "Stop Loss",
        format_price(stop_loss),
    )

with r3:
    st.metric(
        "Take Profit",
        format_price(take_profit),
    )

with r4:
    st.metric(
        "Risk / Reward",
        f"1 : {risk_reward:.2f}",
    )

st.caption(
    "Stop Loss dihitung dari uncertainty model. "
    "Take Profit mengikuti target Risk/Reward. "
    "Perhitungan ini adalah alat bantu analisis, "
    "bukan instruksi trading otomatis."
)


# ============================================================
# GPR SIGNAL COMPONENTS
# ============================================================

st.markdown(
    '<div class="section-title">🔬 GPR Signal Components</div>',
    unsafe_allow_html=True,
)

components = pd.DataFrame(
    [
        [
            "Forecast",
            f"{selected['GPR RETURN']:+.2f}%",
        ],
        [
            "Uncertainty",
            f"{selected['UNCERTAINTY']:.2f}%",
        ],
        [
            "Forecast / Uncertainty",
            f"{abs(selected['GPR RETURN']) / max(selected['UNCERTAINTY'], 0.01):.2f}x",
        ],
        [
            "Signal Agreement",
            f"{selected['SIGNAL AGREEMENT']:.1f}%",
        ],
        [
            "Confidence",
            f"{selected['CONFIDENCE']:.1f}%",
        ],
        [
            "Confidence Label",
            selected["CONF LABEL"],
        ],
        [
            "Momentum 6H",
            f"{selected['MOM 6H']:+.2f}%",
        ],
        [
            "Momentum 12H",
            f"{selected['MOM 12H']:+.2f}%",
        ],
        [
            "Momentum 24H",
            f"{selected['MOM 24H']:+.2f}%",
        ],
        [
            "RSI 14",
            f"{selected['RSI']:.1f}",
        ],
        [
            "Volume Z",
            f"{selected['VOL Z']:+.2f}",
        ],
        [
            "EMA12 Distance",
            f"{selected['EMA12 DIST']:+.2f}%",
        ],
        [
            "EMA24 Distance",
            f"{selected['EMA24 DIST']:+.2f}%",
        ],
    ],
    columns=["Component", "Value"],
)

st.dataframe(
    components,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">📝 Interpretasi</div>',
    unsafe_allow_html=True,
)

direction_word = (
    "bullish"
    if signal == "LONG"
    else "bearish"
)

trend_text = (
    "mendukung"
    if (
        (
            signal == "LONG"
            and selected["EMA12 DIST"] > 0
            and selected["EMA24 DIST"] > 0
        )
        or
        (
            signal == "SHORT"
            and selected["EMA12 DIST"] < 0
            and selected["EMA24 DIST"] < 0
        )
    )
    else "belum sepenuhnya mendukung"
)

st.write(
    f"**{selected_coin}** memiliki "
    f"GPR Score **{selected['GPR SCORE']:.1f}** "
    f"dengan Confidence **{selected['CONFIDENCE']:.1f}%**. "
    f"Berdasarkan kombinasi forecast, momentum, trend EMA, "
    f"RSI, volume dan agreement, bias utama saat ini adalah "
    f"**{direction_word} / {signal}**."
)

st.write(
    f"Trend EMA {timeframe} "
    f"**{trend_text}** arah signal."
)

if selected["CONFIDENCE"] >= 85:

    st.success(
        "Confidence sangat kuat. Namun tetap perhatikan "
        "uncertainty dan kondisi market sebelum mengambil keputusan."
    )

elif selected["CONFIDENCE"] >= 75:

    st.info(
        "Confidence kuat. Signal relatif konsisten, "
        "tetapi masih perlu konfirmasi pergerakan harga."
    )

elif selected["CONFIDENCE"] >= 60:

    st.warning(
        "Confidence cukup. Jangan menganggap signal ini "
        "sebagai kepastian arah harga."
    )

else:

    st.warning(
        "Confidence rendah. Signal sebaiknya dianggap "
        "sebagai kandidat lemah."
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

st.divider()

with st.expander(
    f"📋 ERROR / DIAGNOSTIC ({len(errors)})",
    expanded=False,
):

    if not errors:

        st.success(
            "Tidak ada error pada coin yang dianalisis."
        )

    else:

        error_df = pd.DataFrame(errors)

        st.dataframe(
            error_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GPR Crypto Futures Scanner · "
    "USDT Perpetual · "
    f"{timeframe} · "
    f"Horizon {horizon} candle · "
    "OKX Public Market Data"
)
