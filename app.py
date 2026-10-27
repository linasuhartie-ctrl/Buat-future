import streamlit as st
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ============================================================
# GPR CRYPTO FUTURES SCANNER
# ============================================================
# Public OKX market data - NO API KEY
# Timeframe : 15m / 1H
# Product   : USDT Perpetual / SWAP
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

OKX_BASE = "https://www.okx.com"
CANDLES_URL = f"{OKX_BASE}/api/v5/market/candles"

# Static universe.
# Kita sengaja tidak bergantung pada /public/instruments.
UNIVERSE = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "ADA",
    "SUI",
    "LINK",
    "AVAX",
    "DOT",
    "LTC",
    "BCH",
    "TRX",
    "TON",
    "WLD",
    "APT",
    "ARB",
    "OP",
    "NEAR",
    "ATOM",
    "FIL",
    "ETC",
    "UNI",
    "AAVE",
    "MKR",
    "INJ",
    "SEI",
    "TIA",
    "JUP",
    "JTO",
    "RUNE",
    "KAS",
    "STX",
    "IMX",
    "APE",
    "SAND",
    "MANA",
    "GALA",
    "AXS",
    "FLOW",
    "EOS",
    "XTZ",
    "ALGO",
    "EGLD",
    "THETA",
    "HBAR",
    "ICP",
    "CRV",
    "DYDX",
    "SNX",
    "COMP",
    "LDO",
    "ENS",
    "WIF",
    "BONK",
    "PEPE",
    "SHIB",
    "FLOKI",
    "MANTA",
    "MINA",
    "DYM",
    "STRK",
    "ZK",
    "ZRO",
    "ENA",
    "ONDO",
    "PYTH",
    "TAO",
    "FET",
    "RENDER",
    "AR",
    "KERNEL",
    "MOVE",
    "BERA",
    "KAITO",
    "PENDLE",
    "OM",
    "JASMY",
]

UNIVERSE = list(dict.fromkeys(UNIVERSE))

TIMEFRAMES = {
    "15m": "15m",
    "1H": "1H",
}

HORIZONS = {
    "5 candle": 5,
    "10 candle": 10,
    "20 candle": 20,
    "30 candle": 30,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def inst_id(symbol):
    """
    Convert ticker to OKX perpetual instrument ID.
    Example:
        BTC -> BTC-USDT-SWAP
    """
    return f"{symbol}-USDT-SWAP"


def pct(value, digits=2):
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.{digits}f}%"


def fmt_price(value):
    if value is None or pd.isna(value):
        return "—"

    if abs(value) >= 1000:
        return f"{value:,.2f}"

    if abs(value) >= 1:
        return f"{value:,.4f}"

    return f"{value:.6f}"


# ============================================================
# OKX DATA
# ============================================================

def fetch_json(url, params, timeout=15):
    response = requests.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent": "GPR-Crypto-Futures-Scanner/1.0"
        },
    )

    response.raise_for_status()

    return response.json()


@st.cache_data(ttl=45, show_spinner=False)
def fetch_candles(symbol, timeframe, limit=180):
    """
    Get public OKX candles.

    OKX candle format:
    [timestamp, open, high, low, close,
     volume, volCcy, volCcyQuote, confirm]
    """

    params = {
        "instId": inst_id(symbol),
        "bar": timeframe,
        "limit": str(min(limit, 300)),
    }

    payload = fetch_json(
        CANDLES_URL,
        params,
    )

    if payload.get("code") != "0":
        raise RuntimeError(
            f"OKX code={payload.get('code')} "
            f"msg={payload.get('msg', 'unknown')}"
        )

    rows = payload.get("data", [])

    if not rows:
        raise RuntimeError(
            "OKX tidak mengembalikan candle."
        )

    columns = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol_ccy",
        "vol_quote",
        "confirm",
    ]

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vol_ccy",
        "vol_quote",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["ts"] = pd.to_numeric(
        df["ts"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "ts",
            "close",
        ]
    )

    df = df.sort_values(
        "ts"
    ).reset_index(
        drop=True
    )

    # Jangan memakai candle yang masih berjalan.
    if len(df) > 3:
        if str(df.iloc[-1]["confirm"]) != "1":
            df = df.iloc[:-1]

    return df.reset_index(drop=True)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_ema(series, period):
    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    rsi_value = 100 - (
        100 / (1 + rs)
    )

    return rsi_value.fillna(50)


def calculate_features(df):
    data = df.copy()

    close = data["close"]

    # EMA
    data["ema12"] = calculate_ema(
        close,
        12,
    )

    data["ema24"] = calculate_ema(
        close,
        24,
    )

    # RSI
    data["rsi14"] = calculate_rsi(
        close,
        14,
    )

    # Momentum
    data["mom6"] = (
        close.pct_change(6) * 100
    )

    data["mom12"] = (
        close.pct_change(12) * 100
    )

    data["mom24"] = (
        close.pct_change(24) * 100
    )

    # Volume Z Score
    volume_mean = (
        data["volume"]
        .rolling(30)
        .mean()
    )

    volume_std = (
        data["volume"]
        .rolling(30)
        .std(ddof=0)
        .replace(0, np.nan)
    )

    data["vol_z"] = (
        (data["volume"] - volume_mean)
        / volume_std
    )

    data["vol_z"] = (
        data["vol_z"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    # EMA distance
    data["ema12_dist"] = (
        (close / data["ema12"]) - 1
    ) * 100

    data["ema24_dist"] = (
        (close / data["ema24"]) - 1
    ) * 100

    # True Range
    tr1 = (
        data["high"]
        - data["low"]
    )

    tr2 = (
        data["high"]
        - close.shift(1)
    ).abs()

    tr3 = (
        data["low"]
        - close.shift(1)
    ).abs()

    data["tr"] = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    # ATR %
    data["atr_pct"] = (
        data["tr"]
        .rolling(14)
        .mean()
        / close
    ) * 100

    return data


# ============================================================
# GPR-STYLE MODEL
# ============================================================

def calculate_model(df, horizon):
    """
    GPR-style transparent scoring engine.

    Output:
        signal
        score
        forecast
        uncertainty
        confidence
        confidence_ratio
        agreement
        historical_validation

    IMPORTANT:
    Confidence bukan probabilitas profit.
    """

    if len(df) < 80:
        raise RuntimeError(
            "Data candle tidak cukup."
        )

    x = df.iloc[-1]

    # --------------------------------------------------------
    # 1. MOMENTUM COMPONENT
    # --------------------------------------------------------

    momentum_component = np.tanh(
        (
            0.45 * x["mom6"]
            + 0.35 * x["mom12"]
            + 0.20 * x["mom24"]
        ) / 2.0
    )

    # --------------------------------------------------------
    # 2. TREND COMPONENT
    # --------------------------------------------------------

    trend_component = np.tanh(
        (
            0.65 * x["ema12_dist"]
            + 0.35 * x["ema24_dist"]
        ) / 0.50
    )

    # --------------------------------------------------------
    # 3. RSI COMPONENT
    # --------------------------------------------------------

    rsi_component = np.clip(
        (x["rsi14"] - 50) / 25,
        -1,
        1,
    )

    # --------------------------------------------------------
    # 4. VOLUME COMPONENT
    # --------------------------------------------------------

    volume_component = np.clip(
        x["vol_z"] / 2.5,
        -1,
        1,
    )

    # --------------------------------------------------------
    # 5. HISTORICAL VALIDATION
    # --------------------------------------------------------

    historical_data = df.iloc[
        :-horizon
    ].copy()

    if len(historical_data) < 50:
        historical_validation = 0.0

    else:
        future_return = (
            historical_data["close"]
            .shift(-horizon)
            / historical_data["close"]
            - 1
        )

        historical_condition = (
            0.45
            * np.tanh(
                historical_data["mom6"]
                .fillna(0)
                / 2.0
            )
            + 0.35
            * np.tanh(
                historical_data["ema12_dist"]
                .fillna(0)
                / 0.50
            )
            + 0.20
            * np.clip(
                (
                    historical_data["rsi14"]
                    .fillna(50)
                    - 50
                ) / 25,
                -1,
                1,
            )
        )

        model_direction = np.sign(
            historical_condition
        )

        actual_direction = np.sign(
            future_return
        )

        valid = (
            future_return.notna()
            & (model_direction != 0)
        )

        if valid.sum() < 20:
            historical_validation = 0.0
        else:
            historical_validation = float(
                (
                    actual_direction[valid]
                    == model_direction[valid]
                ).mean()
            )

    # --------------------------------------------------------
    # 6. DIRECTION SCORE
    # --------------------------------------------------------

    direction_score = (
        0.42 * momentum_component
        + 0.33 * trend_component
        + 0.15 * rsi_component
        + 0.10 * volume_component
    )

    if direction_score >= 0:
        signal = "LONG"
    else:
        signal = "SHORT"

    # --------------------------------------------------------
    # 7. FORECAST
    # --------------------------------------------------------

    atr = x["atr_pct"]

    if pd.isna(atr):
        atr = 0.20

    atr = max(
        float(atr),
        0.05,
    )

    raw_forecast = (
        direction_score
        * atr
        * np.sqrt(max(horizon, 1))
        * 1.35
    )

    validation_factor = (
        0.70
        + 0.60 * historical_validation
    )

    forecast = (
        abs(raw_forecast)
        * validation_factor
    )

    forecast = float(
        np.clip(
            forecast,
            0.05,
            15.0,
        )
    )

    if signal == "SHORT":
        forecast *= -1

    # --------------------------------------------------------
    # 8. UNCERTAINTY
    # --------------------------------------------------------

    uncertainty = float(
        max(
            0.08,
            atr
            * (
                0.75
                + 0.80
                * (
                    1
                    - abs(direction_score)
                )
            ),
        )
    )

    # --------------------------------------------------------
    # 9. CONFIDENCE
    # --------------------------------------------------------

    strength = abs(
        direction_score
    )

    confidence = (
        35
        + 38 * strength
        + 20 * historical_validation
        + 7 * min(
            abs(volume_component),
            1,
        )
    )

    confidence = float(
        np.clip(
            confidence,
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # 10. SIGNAL AGREEMENT
    # --------------------------------------------------------

    components = np.array(
        [
            np.sign(
                momentum_component
            ),
            np.sign(
                trend_component
            ),
            np.sign(
                rsi_component
            ),
            np.sign(
                volume_component
            ),
        ]
    )

    target = (
        1
        if signal == "LONG"
        else -1
    )

    agreement = float(
        (
            components == target
        ).mean()
        * 100
    )

    # --------------------------------------------------------
    # 11. CONFIDENCE RATIO
    # --------------------------------------------------------

    confidence_ratio = (
        abs(forecast)
        / max(
            uncertainty,
            1e-9,
        )
    )

    # --------------------------------------------------------
    # 12. GPR SCORE
    # --------------------------------------------------------

    score = 100 * (
        0.42 * strength
        + 0.28 * historical_validation
        + 0.18 * (
            agreement / 100
        )
        + 0.12 * min(
            confidence_ratio / 4.0,
            1.0,
        )
    )

    score = float(
        np.clip(
            score,
            0,
            100,
        )
    )

    return {
        "signal": signal,
        "score": score,
        "forecast": forecast,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "confidence_ratio": confidence_ratio,
        "agreement": agreement,
        "momentum_component": float(
            momentum_component
        ),
        "trend_component": float(
            trend_component
        ),
        "rsi_component": float(
            rsi_component
        ),
        "volume_component": float(
            volume_component
        ),
        "historical_validation": historical_validation,
        "rsi": float(
            x["rsi14"]
        ),
        "price": float(
            x["close"]
        ),
        "mom6": float(
            x["mom6"]
        ),
        "mom12": float(
            x["mom12"]
        ),
        "mom24": float(
            x["mom24"]
        ),
        "vol_z": float(
            x["vol_z"]
        ),
        "ema12_dist": float(
            x["ema12_dist"]
        ),
        "ema24_dist": float(
            x["ema24_dist"]
        ),
        "atr_pct": atr,
        "timestamp": int(
            x["ts"]
        ),
    }


# ============================================================
# 1H CONFIRMATION
# ============================================================

def get_1h_confirmation(symbol):

    try:

        df = fetch_candles(
            symbol,
            "1H",
            120,
        )

        data = calculate_features(
            df
        )

        x = data.iloc[-1]

        confirmation_score = (
            0.50
            * np.tanh(
                x["mom6"] / 2.5
            )
            + 0.30
            * np.tanh(
                x["ema12_dist"] / 0.8
            )
            + 0.20
            * np.clip(
                (
                    x["rsi14"]
                    - 50
                ) / 25,
                -1,
                1,
            )
        )

        if confirmation_score > 0.05:
            return "BULLISH"

        if confirmation_score < -0.05:
            return "BEARISH"

        return "NEUTRAL"

    except Exception:
        return "UNKNOWN"


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_symbol(
    symbol,
    timeframe,
    horizon,
    use_1h_confirmation,
):

    try:

        df = fetch_candles(
            symbol,
            timeframe,
            180,
        )

        data = calculate_features(
            df
        )

        result = calculate_model(
            data,
            horizon,
        )

        result["symbol"] = symbol

        # ----------------------------------------------------
        # 1H CONFIRMATION
        # ----------------------------------------------------

        if use_1h_confirmation:
            confirmation = (
                get_1h_confirmation(
                    symbol
                )
            )
        else:
            confirmation = "OFF"

        result[
            "confirmation_1h"
        ] = confirmation

        # ----------------------------------------------------
        # CONFIRMATION BONUS / PENALTY
        # ----------------------------------------------------

        if confirmation in (
            "BULLISH",
            "BEARISH",
        ):

            aligned = (
                (
                    result["signal"]
                    == "LONG"
                )
                and (
                    confirmation
                    == "BULLISH"
                )
            ) or (
                (
                    result["signal"]
                    == "SHORT"
                )
                and (
                    confirmation
                    == "BEARISH"
                )
            )

            if aligned:

                result["score"] = min(
                    100,
                    result["score"]
                    + 8,
                )

                result[
                    "confidence"
                ] = min(
                    100,
                    result[
                        "confidence"
                    ]
                    + 7,
                )

            else:

                result["score"] = max(
                    0,
                    result["score"]
                    - 8,
                )

                result[
                    "confidence"
                ] = max(
                    0,
                    result[
                        "confidence"
                    ]
                    - 7,
                )

        return result, None

    except Exception as error:

        return (
            None,
            f"{type(error).__name__}: {error}",
        )


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_risk_plan(
    row,
    reward_ratio,
):

    entry = float(
        row["price"]
    )

    uncertainty = max(
        float(
            row["uncertainty"]
        ) / 100,
        0.0005,
    )

    stop_distance = (
        entry
        * uncertainty
    )

    if row["signal"] == "LONG":

        stop_loss = (
            entry
            - stop_distance
        )

        take_profit = (
            entry
            + stop_distance
            * reward_ratio
        )

    else:

        stop_loss = (
            entry
            + stop_distance
        )

        take_profit = (
            entry
            - stop_distance
            * reward_ratio
        )

    return (
        entry,
        stop_loss,
        take_profit,
        reward_ratio,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Scanner Settings"
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    list(
        TIMEFRAMES.keys()
    ),
    index=0,
)

horizon_label = st.sidebar.selectbox(
    "Horizon GPR",
    list(
        HORIZONS.keys()
    ),
    index=1,
)

horizon = HORIZONS[
    horizon_label
]

coin_count = st.sidebar.selectbox(
    "Jumlah coin",
    [20, 40, 60, 79],
    index=3,
)

min_score = st.sidebar.slider(
    "Minimum GPR Score",
    40,
    95,
    70,
    1,
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence (%)",
    40,
    95,
    60,
    1,
)

workers = st.sidebar.slider(
    "Parallel workers",
    1,
    10,
    5,
    1,
)

use_1h_confirmation = (
    st.sidebar.checkbox(
        "Gunakan konfirmasi 1H",
        value=True,
        help=(
            "Untuk scanner 15m, "
            "arah 1H digunakan sebagai "
            "konfirmasi tambahan."
        ),
    )
)

scan_button = st.sidebar.button(
    "🚀 SCAN FUTURES",
    type="primary",
    use_container_width=True,
)

clear_button = st.sidebar.button(
    "🗑️ CLEAR CACHE",
    use_container_width=True,
)

if clear_button:

    st.cache_data.clear()

    st.session_state.pop(
        "results",
        None,
    )

    st.session_state.pop(
        "errors",
        None,
    )

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 GPR CRYPTO FUTURES SCANNER"
)

st.caption(
    f"Target utama: USDT Perpetual · "
    f"Timeframe {timeframe} · "
    f"GPR sebagai mesin ranking utama"
)

st.info(
    "Mode data: public market data, tanpa API key. "
    f"Provider OKX · {timeframe} · "
    f"Horizon {horizon} candle."
)


# ============================================================
# SCAN
# ============================================================

if (
    scan_button
    or "results"
    not in st.session_state
):

    selected_symbols = (
        UNIVERSE[:coin_count]
    )

    results = []
    errors = []

    progress = st.progress(
        0
    )

    status_text = st.empty()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {}

        for symbol in selected_symbols:

            future = executor.submit(
                analyze_symbol,
                symbol,
                timeframe,
                horizon,
                use_1h_confirmation,
            )

            futures[
                future
            ] = symbol

        total = len(
            futures
        )

        completed = 0

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            try:

                result, error = (
                    future.result()
                )

            except Exception as error:

                result = None

                error = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

            if result is not None:

                results.append(
                    result
                )

            else:

                errors.append(
                    {
                        "Symbol": symbol,
                        "Status": error,
                    }
                )

            completed += 1

            progress.progress(
                completed
                / total
            )

            status_text.text(
                f"Scanning "
                f"{completed}/{total}: "
                f"{symbol}"
            )

    progress.empty()

    status_text.empty()

    results_df = pd.DataFrame(
        results
    )

    errors_df = pd.DataFrame(
        errors
    )

    if not results_df.empty:

        results_df = (
            results_df
            .sort_values(
                [
                    "score",
                    "confidence",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        results_df[
            "rank"
        ] = (
            np.arange(
                1,
                len(results_df)
                + 1,
            )
        )

    st.session_state[
        "results"
    ] = results_df

    st.session_state[
        "errors"
    ] = errors_df

    st.session_state[
        "scan_time"
    ] = datetime.now(
        timezone.utc
    )


# ============================================================
# LOAD RESULT
# ============================================================

results_df = (
    st.session_state.get(
        "results",
        pd.DataFrame(),
    )
)

errors_df = (
    st.session_state.get(
        "errors",
        pd.DataFrame(),
    )
)

if results_df.empty:

    st.error(
        "Belum ada hasil scan."
    )

    st.stop()


# ============================================================
# CANDIDATES
# ============================================================

candidates = (
    results_df[
        (
            results_df[
                "score"
            ]
            >= min_score
        )
        & (
            results_df[
                "confidence"
            ]
            >= min_confidence
        )
    ]
    .copy()
)


# ============================================================
# TOP SUMMARY
# ============================================================

best = results_df.iloc[0]

col1, col2, col3, col4 = (
    st.columns(4)
)

col1.metric(
    "UNIVERSE",
    len(results_df),
    "USDT Perpetual",
)

col2.metric(
    "TIMEFRAME",
    timeframe,
    "Primary timeframe",
)

hours = (
    horizon * 0.25
    if timeframe == "15m"
    else horizon
)

col3.metric(
    "HORIZON",
    f"{horizon} candle",
    f"≈ {hours:g} jam",
)

col4.metric(
    "PROVIDER",
    "OKX",
    "Public market data",
)


st.divider()


col1, col2, col3, col4 = (
    st.columns(4)
)

col1.metric(
    "BERHASIL DIANALISIS",
    len(results_df),
)

col2.metric(
    f"LOLOS SCORE ≥ {min_score}",
    len(candidates),
)

col3.metric(
    "KANDIDAT",
    min(
        len(candidates),
        10,
    ),
)

col4.metric(
    "GPR RETURN TERTINGGI",
    pct(
        best["forecast"]
    ),
    f"Score {best['score']:.1f}",
)


# ============================================================
# RANKING
# ============================================================

st.divider()

st.subheader(
    "🏆 GPR Futures Ranking"
)

display_columns = [
    "rank",
    "symbol",
    "signal",
    "score",
    "forecast",
    "uncertainty",
    "confidence",
    "confidence_ratio",
    "price",
    "rsi",
    "mom6",
    "mom12",
    "mom24",
    "vol_z",
]

ranking = (
    results_df[
        display_columns
    ]
    .copy()
)

ranking.columns = [
    "Rank",
    "Coin",
    "Signal",
    "GPR Score",
    "GPR Return",
    "Uncertainty",
    "Confidence %",
    "Conf Ratio",
    "Price",
    "RSI",
    "Mom 6H",
    "Mom 12H",
    "Mom 24H",
    "Vol Z",
]


ranking[
    "GPR Score"
] = ranking[
    "GPR Score"
].map(
    lambda x: f"{x:.1f}"
)

ranking[
    "GPR Return"
] = ranking[
    "GPR Return"
].map(
    lambda x: f"{x:+.2f}%"
)

ranking[
    "Uncertainty"
] = ranking[
    "Uncertainty"
].map(
    lambda x: f"{x:.2f}%"
)

ranking[
    "Confidence %"
] = ranking[
    "Confidence %"
].map(
    lambda x: f"{x:.0f}%"
)

ranking[
    "Conf Ratio"
] = ranking[
    "Conf Ratio"
].map(
    lambda x: f"{x:.2f}x"
)

ranking[
    "Price"
] = ranking[
    "Price"
].map(
    fmt_price
)

ranking[
    "RSI"
] = ranking[
    "RSI"
].map(
    lambda x: f"{x:.1f}"
)

ranking[
    "Mom 6H"
] = ranking[
    "Mom 6H"
].map(
    lambda x: f"{x:+.2f}%"
)

ranking[
    "Mom 12H"
] = ranking[
    "Mom 12H"
].map(
    lambda x: f"{x:+.2f}%"
)

ranking[
    "Mom 24H"
] = ranking[
    "Mom 24H"
].map(
    lambda x: f"{x:+.2f}%"
)

ranking[
    "Vol Z"
] = ranking[
    "Vol Z"
].map(
    lambda x: f"{x:+.2f}"
)

st.dataframe(
    ranking,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TOP SIGNALS
# ============================================================

st.divider()

st.subheader(
    "🎯 Top GPR Signals"
)

top_n = min(
    6,
    len(candidates),
)

if top_n == 0:

    st.warning(
        "Belum ada kandidat yang memenuhi "
        "Minimum Score dan Minimum Confidence. "
        "Turunkan threshold jika ingin melihat "
        "lebih banyak kandidat."
    )

else:

    card_columns = st.columns(2)

    for index, (
        _,
        row,
    ) in enumerate(
        candidates.head(
            top_n
        ).iterrows()
    ):

        with card_columns[
            index % 2
        ]:

            if row[
                "signal"
            ] == "LONG":

                direction = "🟢 LONG"

            else:

                direction = "🔴 SHORT"

            st.markdown(
                f"### {row['symbol']}"
            )

            st.write(
                f"**{direction}**"
            )

            a, b = (
                st.columns(2)
            )

            a.metric(
                "GPR Score",
                f"{row['score']:.1f}",
            )

            b.metric(
                "Confidence",
                f"{row['confidence']:.0f}%",
            )

            a, b = (
                st.columns(2)
            )

            a.metric(
                "GPR Return",
                pct(
                    row["forecast"]
                ),
            )

            b.metric(
                "Uncertainty",
                f"{row['uncertainty']:.2f}%",
            )

            st.caption(
                f"Confidence Ratio "
                f"{row['confidence_ratio']:.2f}x · "
                f"Agreement "
                f"{row['agreement']:.0f}% · "
                f"1H "
                f"{row['confirmation_1h']}"
            )


# ============================================================
# DETAIL ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "🔎 Detail Analysis"
)

detail_symbols = (
    candidates[
        "symbol"
    ].tolist()
)

if not detail_symbols:

    detail_symbols = (
        results_df[
            "symbol"
        ].tolist()
    )

selected_symbol = st.selectbox(
    "Pilih coin",
    detail_symbols,
)

selected_row = (
    results_df[
        results_df[
            "symbol"
        ]
        == selected_symbol
    ]
    .iloc[0]
)


# ============================================================
# DETAIL HEADER
# ============================================================

a, b, c, d = (
    st.columns(4)
)

a.metric(
    "SIGNAL",
    selected_row[
        "signal"
    ],
)

b.metric(
    "GPR SCORE",
    f"{selected_row['score']:.1f}",
)

c.metric(
    "FORECAST",
    pct(
        selected_row[
            "forecast"
        ]
    ),
)

d.metric(
    "CONFIDENCE",
    f"{selected_row['confidence']:.0f}%",
)


# ============================================================
# CONFIDENCE
# ============================================================

st.markdown(
    "#### 📊 Cara Membaca Confidence"
)

confidence = float(
    selected_row[
        "confidence"
    ]
)

confidence_ratio = float(
    selected_row[
        "confidence_ratio"
    ]
)

if confidence >= 80:

    confidence_label = (
        "SANGAT KUAT"
    )

elif confidence >= 70:

    confidence_label = (
        "KUAT"
    )

elif confidence >= 60:

    confidence_label = (
        "CUKUP"
    )

elif confidence >= 50:

    confidence_label = (
        "LEMAH"
    )

else:

    confidence_label = (
        "SANGAT LEMAH"
    )


st.info(
    f"**{confidence_label} — "
    f"Confidence {confidence:.0f}%**. "
    "Confidence menunjukkan seberapa kuat "
    "model mendukung arah sinyal berdasarkan "
    "kombinasi momentum, trend, RSI, volume "
    "dan validasi historis. "
    "Ini bukan probabilitas pasti profit. "
    f"Confidence Ratio = "
    f"{confidence_ratio:.2f}x."
)


a, b, c, d = (
    st.columns(4)
)

a.metric(
    "Confidence",
    f"{confidence:.0f}%",
)

b.metric(
    "Confidence Ratio",
    f"{confidence_ratio:.2f}x",
)

c.metric(
    "Signal Agreement",
    f"{selected_row['agreement']:.0f}%",
)

d.metric(
    "Historical Validation",
    f"{selected_row['historical_validation'] * 100:.0f}%",
)


st.progress(
    int(
        np.clip(
            confidence,
            0,
            100,
        )
    )
)


# ============================================================
# TECHNICAL DATA
# ============================================================

st.markdown(
    "#### 📐 Technical Data"
)

a, b, c, d = (
    st.columns(4)
)

a.metric(
    "Harga",
    fmt_price(
        selected_row[
            "price"
        ]
    ),
)

b.metric(
    "RSI 14",
    f"{selected_row['rsi']:.1f}",
)

c.metric(
    "Momentum 6H",
    pct(
        selected_row[
            "mom6"
        ]
    ),
)

d.metric(
    "Momentum 12H",
    pct(
        selected_row[
            "mom12"
        ]
    ),
)


a, b, c, d = (
    st.columns(4)
)

a.metric(
    "Momentum 24H",
    pct(
        selected_row[
            "mom24"
        ]
    ),
)

b.metric(
    "Volume Z",
    f"{selected_row['vol_z']:+.2f}",
)

c.metric(
    "EMA12 Distance",
    pct(
        selected_row[
            "ema12_dist"
        ]
    ),
)

d.metric(
    "EMA24 Distance",
    pct(
        selected_row[
            "ema24_dist"
        ]
    ),
)


last_candle = datetime.fromtimestamp(
    selected_row[
        "timestamp"
    ] / 1000,
    tz=timezone.utc,
).strftime(
    "%Y-%m-%d %H:%M UTC"
)

st.caption(
    f"Last confirmed candle: "
    f"{last_candle}"
)


# ============================================================
# RISK MANAGEMENT
# ============================================================

st.divider()

st.subheader(
    "🎯 Risk Management"
)

reward_ratio = st.slider(
    "Risk / Reward Target",
    1.00,
    3.00,
    1.67,
    0.01,
)

entry, stop_loss, take_profit, rr = (
    calculate_risk_plan(
        selected_row,
        reward_ratio,
    )
)


a, b, c, d = (
    st.columns(4)
)

a.metric(
    "Entry / Current",
    fmt_price(entry),
)

b.metric(
    "Stop Loss",
    fmt_price(stop_loss),
)

c.metric(
    "Take Profit",
    fmt_price(take_profit),
)

d.metric(
    "Risk / Reward",
    f"1 : {rr:.2f}",
)


st.caption(
    "Stop Loss dihitung dari uncertainty model. "
    "Take Profit mengikuti target Risk/Reward. "
    "Perhitungan ini adalah alat bantu analisis, "
    "bukan instruksi trading otomatis."
)


# ============================================================
# SIGNAL COMPONENTS
# ============================================================

st.markdown(
    "#### 🧪 GPR Signal Components"
)

components = pd.DataFrame(
    {
        "Component": [
            "Momentum",
            "Trend EMA",
            "RSI",
            "Volume",
            "Historical validation",
            "Signal agreement",
            "1H confirmation",
        ],
        "Value": [
            selected_row[
                "momentum_component"
            ],
            selected_row[
                "trend_component"
            ],
            selected_row[
                "rsi_component"
            ],
            selected_row[
                "volume_component"
            ],
            selected_row[
                "historical_validation"
            ],
            selected_row[
                "agreement"
            ] / 100,
            selected_row[
                "confirmation_1h"
            ],
        ],
    }
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
    "#### 📝 Interpretasi"
)

symbol = selected_row[
    "symbol"
]

signal = selected_row[
    "signal"
]

score = selected_row[
    "score"
]

forecast = selected_row[
    "forecast"
]

confirmation = selected_row[
    "confirmation_1h"
]


st.write(
    f"**{symbol}** memiliki GPR Score "
    f"**{score:.1f}** dengan Confidence "
    f"**{confidence:.0f}%**. "
    f"Bias model saat ini adalah "
    f"**{signal}** dengan estimasi "
    f"forecast **{pct(forecast)}**."
)


if confirmation != "UNKNOWN":

    if signal == "LONG" and confirmation == "BULLISH":

        st.success(
            "Konfirmasi 1H BULLISH dan "
            "searah dengan sinyal 15m."
        )

    elif signal == "SHORT" and confirmation == "BEARISH":

        st.success(
            "Konfirmasi 1H BEARISH dan "
            "searah dengan sinyal 15m."
        )

    elif confirmation == "NEUTRAL":

        st.warning(
            "Konfirmasi 1H masih NEUTRAL. "
            "Sinyal 15m sebaiknya dianggap "
            "belum mendapat dukungan timeframe besar."
        )

    else:

        st.warning(
            f"Sinyal {signal} pada {timeframe} "
            f"belum searah dengan konfirmasi 1H "
            f"({confirmation})."
        )


# ============================================================
# DIAGNOSTIC
# ============================================================

st.divider()

with st.expander(
    f"📋 ERROR / DIAGNOSTIC ({len(errors_df)})"
):

    if errors_df.empty:

        st.success(
            "Tidak ada error."
        )

    else:

        st.dataframe(
            errors_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GPR Crypto Futures Scanner · "
    "OKX Public Market Data · "
    "15m / 1H · Research Prototype · "
    "Tidak melakukan order otomatis."
)
