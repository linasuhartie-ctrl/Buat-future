import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

# ============================================================
# GPR CRYPTO FUTURES SCANNER
# ============================================================
# Public OKX market data - NO API KEY
# Timeframe : 15m / 1H
# Product   : USDT Perpetual / SWAP
#
# RANKING UTAMA:
#   Final Trading Score
#
# VALIDATION:
#   Forward historical validation sesuai horizon
#   5 candle / 10 candle / 20 candle
#
# CATATAN:
#   Confidence bukan probabilitas profit.
#   Scanner tidak melakukan order otomatis.
# ============================================================


# ============================================================
# PAGE CONFIG
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

CANDLES_URL = (
    f"{OKX_BASE}/api/v5/market/candles"
)


# ============================================================
# STATIC UNIVERSE
# ============================================================

UNIVERSE = list(dict.fromkeys([
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
]))


# ============================================================
# TIMEFRAME
# ============================================================

TIMEFRAMES = {
    "15m": "15m",
    "1H": "1H",
}


# ============================================================
# HORIZON
# ============================================================

HORIZONS = {
    "5 candle": 5,
    "10 candle": 10,
    "20 candle": 20,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def inst_id(symbol):
    return f"{symbol}-USDT-SWAP"


def pct(value, digits=2):

    if value is None:
        return "—"

    if pd.isna(value):
        return "—"

    return f"{value:+.{digits}f}%"


def fmt_price(value):

    if value is None:
        return "—"

    if pd.isna(value):
        return "—"

    if abs(value) >= 1000:
        return f"{value:,.2f}"

    if abs(value) >= 1:
        return f"{value:,.4f}"

    return f"{value:.6f}"


def timeframe_bars_for_hours(
    hours,
    timeframe
):

    if timeframe == "15m":

        minutes_per_candle = 15

    else:

        minutes_per_candle = 60

    return max(
        1,
        int(
            round(
                hours
                * 60
                / minutes_per_candle
            )
        )
    )


# ============================================================
# OKX DATA
# ============================================================

@st.cache_data(
    ttl=45,
    show_spinner=False
)
def fetch_candles(
    symbol,
    timeframe,
    limit=300
):

    params = {
        "instId": inst_id(symbol),
        "bar": timeframe,
        "limit": str(
            min(
                limit,
                300
            )
        ),
    }

    response = requests.get(
        CANDLES_URL,
        params=params,
        timeout=15,
        headers={
            "User-Agent":
                "GPR-Crypto-Futures-Scanner/2.0"
        },
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("code") != "0":

        raise RuntimeError(
            f"OKX code={payload.get('code')} "
            f"msg={payload.get('msg', 'unknown')}"
        )

    rows = payload.get(
        "data",
        []
    )

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
        columns=columns
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
            errors="coerce"
        )

    df["ts"] = pd.to_numeric(
        df["ts"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    df = (
        df
        .sort_values("ts")
        .drop_duplicates("ts")
        .reset_index(drop=True)
    )

    # Hanya candle yang sudah confirmed.
    if "confirm" in df.columns:

        df = df[
            df["confirm"]
            .astype(str)
            == "1"
        ].reset_index(drop=True)

    if len(df) < 140:

        raise RuntimeError(
            f"Data candle tidak cukup: {len(df)}"
        )

    return df


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_ema(
    series,
    period
):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def calculate_rsi(
    series,
    period=14
):

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
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    return rsi.fillna(50)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_features(
    df,
    timeframe
):

    x = df.copy()

    close = x["close"]

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    x["ema12"] = calculate_ema(
        close,
        12
    )

    x["ema24"] = calculate_ema(
        close,
        24
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    x["rsi14"] = calculate_rsi(
        close,
        14
    )

    # --------------------------------------------------------
    # TIMEFRAME-AWARE MOMENTUM
    #
    # 15m:
    # 6H  = 24 candle
    # 12H = 48 candle
    # 24H = 96 candle
    #
    # 1H:
    # 6H  = 6 candle
    # 12H = 12 candle
    # 24H = 24 candle
    # --------------------------------------------------------

    bars_6h = (
        timeframe_bars_for_hours(
            6,
            timeframe
        )
    )

    bars_12h = (
        timeframe_bars_for_hours(
            12,
            timeframe
        )
    )

    bars_24h = (
        timeframe_bars_for_hours(
            24,
            timeframe
        )
    )

    x["mom6"] = (
        close.pct_change(
            bars_6h
        )
        * 100
    )

    x["mom12"] = (
        close.pct_change(
            bars_12h
        )
        * 100
    )

    x["mom24"] = (
        close.pct_change(
            bars_24h
        )
        * 100
    )

    # --------------------------------------------------------
    # VOLUME Z SCORE
    # --------------------------------------------------------

    volume_mean = (
        x["volume"]
        .rolling(30)
        .mean()
    )

    volume_std = (
        x["volume"]
        .rolling(30)
        .std(ddof=0)
        .replace(
            0,
            np.nan
        )
    )

    x["vol_z"] = (
        (
            x["volume"]
            - volume_mean
        )
        /
        volume_std
    )

    x["vol_z"] = (
        x["vol_z"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # EMA DISTANCE
    # --------------------------------------------------------

    x["ema12_dist"] = (
        (
            close
            /
            x["ema12"]
        )
        - 1
    ) * 100

    x["ema24_dist"] = (
        (
            close
            /
            x["ema24"]
        )
        - 1
    ) * 100

    # --------------------------------------------------------
    # TRUE RANGE
    # --------------------------------------------------------

    tr1 = (
        x["high"]
        -
        x["low"]
    )

    tr2 = (
        x["high"]
        -
        close.shift(1)
    ).abs()

    tr3 = (
        x["low"]
        -
        close.shift(1)
    ).abs()

    x["tr"] = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1
    ).max(axis=1)

    # --------------------------------------------------------
    # ATR %
    # --------------------------------------------------------

    x["atr_pct"] = (
        x["tr"]
        .rolling(14)
        .mean()
        /
        close
    ) * 100

    return x


# ============================================================
# MODEL COMPONENTS
# ============================================================

def component_values(
    row
):

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = np.tanh(
        (
            0.45 * row["mom6"]
            +
            0.35 * row["mom12"]
            +
            0.20 * row["mom24"]
        )
        / 2.0
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend = np.tanh(
        (
            0.65 * row["ema12_dist"]
            +
            0.35 * row["ema24_dist"]
        )
        / 0.50
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi = np.clip(
        (
            row["rsi14"]
            - 50
        )
        / 25,
        -1,
        1
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume = np.clip(
        row["vol_z"]
        / 2.5,
        -1,
        1
    )

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    direction = (
        0.42 * momentum
        +
        0.33 * trend
        +
        0.15 * rsi
        +
        0.10 * volume
    )

    signal = (
        "LONG"
        if direction >= 0
        else "SHORT"
    )

    return (
        momentum,
        trend,
        rsi,
        volume,
        direction,
        signal
    )


# ============================================================
# HISTORICAL FORWARD VALIDATION
# ============================================================

def historical_validation(
    data,
    horizon,
    lookback=180
):

    """
    Walk-forward validation.

    Untuk setiap candle historis:

        1. Hitung signal hanya dari data sampai candle tersebut.
        2. Lihat return aktual N candle setelahnya.
        3. Cek apakah arah prediksi benar.

    Tidak menggunakan future candle untuk membentuk signal.
    """

    if len(data) <= (
        horizon + 50
    ):

        return {
            "win_rate": 0.50,
            "mean_return": 0.0,
            "median_return": 0.0,
            "samples": 0,
            "recent_win_rate": 0.50,
            "recent_mean_return": 0.0,
        }

    start = max(
        50,
        len(data)
        -
        lookback
        -
        horizon
    )

    end = (
        len(data)
        -
        horizon
    )

    records = []

    for i in range(
        start,
        end
    ):

        row = data.iloc[i]

        if (
            pd.isna(row["mom24"])
            or
            pd.isna(row["atr_pct"])
        ):
            continue

        (
            momentum,
            trend,
            rsi,
            volume,
            direction,
            signal
        ) = component_values(
            row
        )

        # Hindari signal yang terlalu netral.
        if abs(direction) < 0.03:
            continue

        future_return = (
            data.iloc[
                i + horizon
            ]["close"]
            /
            row["close"]
            - 1
        )

        if pd.isna(
            future_return
        ):
            continue

        direction_multiplier = (
            1
            if signal == "LONG"
            else -1
        )

        # Return positif = prediksi benar.
        signed_return = (
            future_return
            *
            direction_multiplier
        )

        records.append(
            (
                i,
                signed_return
            )
        )

    if len(records) < 20:

        return {
            "win_rate": 0.50,
            "mean_return": 0.0,
            "median_return": 0.0,
            "samples": len(records),
            "recent_win_rate": 0.50,
            "recent_mean_return": 0.0,
        }

    values = np.array(
        [
            item[1]
            for item in records
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # FULL HISTORICAL
    # --------------------------------------------------------

    win_rate = float(
        (
            values > 0
        ).mean()
    )

    mean_return = float(
        np.mean(values)
        * 100
    )

    median_return = float(
        np.median(values)
        * 100
    )

    # --------------------------------------------------------
    # RECENT VALIDATION
    # --------------------------------------------------------

    recent_values = (
        values[
            -min(
                50,
                len(values)
            ):
        ]
    )

    recent_win_rate = float(
        (
            recent_values > 0
        ).mean()
    )

    recent_mean_return = float(
        np.mean(
            recent_values
        )
        * 100
    )

    return {
        "win_rate": win_rate,
        "mean_return": mean_return,
        "median_return": median_return,
        "samples": len(values),
        "recent_win_rate": recent_win_rate,
        "recent_mean_return": recent_mean_return,
    }


# ============================================================
# MODEL
# ============================================================

def calculate_model(
    data,
    horizon
):

    if len(data) < 140:

        raise RuntimeError(
            "Data candle tidak cukup."
        )

    row = data.iloc[-1]

    if (
        pd.isna(row["mom24"])
        or
        pd.isna(row["atr_pct"])
    ):

        raise RuntimeError(
            "Feature belum lengkap."
        )

    (
        momentum_component,
        trend_component,
        rsi_component,
        volume_component,
        direction_score,
        signal
    ) = component_values(
        row
    )

    # ========================================================
    # HISTORICAL VALIDATION
    # ========================================================

    validation = historical_validation(
        data,
        horizon
    )

    historical_validation_value = (
        validation["win_rate"]
    )

    recent_validation = (
        validation["recent_win_rate"]
    )

    # ========================================================
    # SIGNAL AGREEMENT
    # ========================================================

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

    # ========================================================
    # ATR
    # ========================================================

    atr = row["atr_pct"]

    if pd.isna(atr):

        atr = 0.20

    atr = max(
        float(atr),
        0.05
    )

    # ========================================================
    # FORECAST
    # ========================================================

    # Forecast teoritis.
    theoretical_forecast = (
        abs(direction_score)
        *
        atr
        *
        np.sqrt(
            max(
                horizon,
                1
            )
        )
        *
        1.35
    )

    # Forecast empiris berdasarkan historical forward return.
    empirical_forecast = max(
        validation[
            "median_return"
        ],
        0.0
    )

    # Blend.
    forecast = (
        0.60
        *
        theoretical_forecast
        +
        0.40
        *
        empirical_forecast
    )

    forecast = float(
        np.clip(
            forecast,
            0.03,
            15.0
        )
    )

    if signal == "SHORT":

        forecast *= -1

    # ========================================================
    # UNCERTAINTY
    # ========================================================

    uncertainty = float(
        max(
            0.08,
            atr
            *
            (
                0.75
                +
                0.80
                *
                (
                    1
                    -
                    abs(
                        direction_score
                    )
                )
            )
        )
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    strength = abs(
        direction_score
    )

    confidence = (
        30
        +
        35 * strength
        +
        20
        *
        historical_validation_value
        +
        10
        *
        (
            agreement
            / 100
        )
        +
        5
        *
        min(
            abs(
                volume_component
            ),
            1
        )
    )

    confidence = float(
        np.clip(
            confidence,
            0,
            100
        )
    )

    # ========================================================
    # CONFIDENCE RATIO
    # ========================================================

    confidence_ratio = (
        abs(forecast)
        /
        max(
            uncertainty,
            1e-9
        )
    )

    # ========================================================
    # SCORE COMPONENTS
    # ========================================================

    strength_score = np.clip(
        strength,
        0,
        1
    )

    edge_score = np.clip(
        (
            historical_validation_value
            - 0.50
        )
        /
        0.25,
        0,
        1
    )

    recent_edge_score = np.clip(
        (
            recent_validation
            - 0.50
        )
        /
        0.25,
        0,
        1
    )

    agreement_score = (
        agreement
        / 100
    )

    forecast_quality = np.clip(
        abs(forecast)
        /
        max(
            uncertainty * 1.5,
            0.08
        ),
        0,
        1
    )

    ratio_score = min(
        confidence_ratio / 3.0,
        1.0
    )

    # ========================================================
    # GPR SCORE
    # ========================================================

    gpr_score = 100 * (
        0.35
        *
        strength_score

        +

        0.25
        *
        historical_validation_value

        +

        0.15
        *
        agreement_score

        +

        0.10
        *
        ratio_score

        +

        0.15
        *
        recent_validation
    )

    gpr_score = float(
        np.clip(
            gpr_score,
            0,
            100
        )
    )

    # ========================================================
    # FINAL TRADING SCORE
    # ========================================================

    # Ini menjadi ranking utama.
    #
    # Tidak hanya bertanya:
    # "Seberapa bagus signal sekarang?"
    #
    # Tetapi:
    # "Seberapa bagus signal + historical edge
    #  + consistency + forecast quality?"
    #

    final_score = 100 * (

        0.25
        *
        strength_score

        +

        0.25
        *
        edge_score

        +

        0.15
        *
        recent_edge_score

        +

        0.15
        *
        agreement_score

        +

        0.10
        *
        forecast_quality

        +

        0.10
        *
        (
            confidence
            / 100
        )
    )

    final_score = float(
        np.clip(
            final_score,
            0,
            100
        )
    )

    return {

        "signal":
            signal,

        "gpr_score":
            gpr_score,

        "final_score":
            final_score,

        "forecast":
            forecast,

        "uncertainty":
            uncertainty,

        "confidence":
            confidence,

        "confidence_ratio":
            confidence_ratio,

        "agreement":
            agreement,

        "historical_validation":
            historical_validation_value,

        "recent_validation":
            recent_validation,

        "validation_samples":
            validation[
                "samples"
            ],

        "historical_mean_return":
            validation[
                "mean_return"
            ],

        "historical_median_return":
            validation[
                "median_return"
            ],

        "recent_mean_return":
            validation[
                "recent_mean_return"
            ],

        "momentum_component":
            float(
                momentum_component
            ),

        "trend_component":
            float(
                trend_component
            ),

        "rsi_component":
            float(
                rsi_component
            ),

        "volume_component":
            float(
                volume_component
            ),

        "rsi":
            float(
                row["rsi14"]
            ),

        "price":
            float(
                row["close"]
            ),

        "mom6":
            float(
                row["mom6"]
            ),

        "mom12":
            float(
                row["mom12"]
            ),

        "mom24":
            float(
                row["mom24"]
            ),

        "vol_z":
            float(
                row["vol_z"]
            ),

        "ema12_dist":
            float(
                row["ema12_dist"]
            ),

        "ema24_dist":
            float(
                row["ema24_dist"]
            ),

        "atr_pct":
            atr,

        "timestamp":
            int(
                row["ts"]
            ),
    }


# ============================================================
# 1H CONFIRMATION
# ============================================================

def get_1h_confirmation(
    symbol
):

    try:

        raw = fetch_candles(
            symbol,
            "1H",
            180
        )

        data = calculate_features(
            raw,
            "1H"
        )

        row = data.iloc[-1]

        confirmation_score = (

            0.50
            *
            np.tanh(
                row["mom6"]
                /
                2.5
            )

            +

            0.30
            *
            np.tanh(
                row["ema12_dist"]
                /
                0.8
            )

            +

            0.20
            *
            np.clip(
                (
                    row["rsi14"]
                    - 50
                )
                /
                25,
                -1,
                1
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
    use_1h_confirmation
):

    try:

        raw = fetch_candles(
            symbol,
            timeframe,
            300
        )

        data = calculate_features(
            raw,
            timeframe
        )

        result = calculate_model(
            data,
            horizon
        )

        result["symbol"] = symbol

        # ====================================================
        # 1H CONFIRMATION
        # ====================================================

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

        # ====================================================
        # 1H BONUS / PENALTY
        # ====================================================

        if confirmation in (
            "BULLISH",
            "BEARISH"
        ):

            aligned = (

                (
                    result["signal"]
                    == "LONG"
                )

                and

                (
                    confirmation
                    == "BULLISH"
                )
            ) or (

                (
                    result["signal"]
                    == "SHORT"
                )

                and

                (
                    confirmation
                    == "BEARISH"
                )
            )

            if aligned:

                result[
                    "final_score"
                ] = min(
                    100,
                    result[
                        "final_score"
                    ]
                    + 4
                )

                result[
                    "confidence"
                ] = min(
                    100,
                    result[
                        "confidence"
                    ]
                    + 4
                )

            else:

                result[
                    "final_score"
                ] = max(
                    0,
                    result[
                        "final_score"
                    ]
                    - 6
                )

                result[
                    "confidence"
                ] = max(
                    0,
                    result[
                        "confidence"
                    ]
                    - 5
                )

        # ====================================================
        # TRADEABILITY
        # ====================================================

        if (

            result[
                "validation_samples"
            ] >= 30

            and

            result[
                "historical_validation"
            ] >= 0.55

            and

            result[
                "recent_validation"
            ] >= 0.52

            and

            abs(
                result["forecast"]
            )
            >=
            result[
                "uncertainty"
            ]
            * 0.75

        ):

            result[
                "trade_status"
            ] = "TRADEABLE"

        elif (

            result[
                "validation_samples"
            ] >= 20

            and

            result[
                "historical_validation"
            ] >= 0.52

        ):

            result[
                "trade_status"
            ] = "WATCH"

        else:

            result[
                "trade_status"
            ] = "NO TRADE"

        return (
            result,
            None
        )

    except Exception as error:

        return (
            None,
            f"{type(error).__name__}: {error}"
        )


# ============================================================
# RISK MANAGEMENT
# ============================================================

def calculate_risk_plan(
    row,
    reward_ratio
):

    entry = float(
        row["price"]
    )

    uncertainty = max(
        float(
            row["uncertainty"]
        )
        / 100,
        0.0005
    )

    stop_distance = (
        entry
        *
        uncertainty
    )

    if row["signal"] == "LONG":

        stop_loss = (
            entry
            -
            stop_distance
        )

        take_profit = (
            entry
            +
            stop_distance
            *
            reward_ratio
        )

    else:

        stop_loss = (
            entry
            +
            stop_distance
        )

        take_profit = (
            entry
            -
            stop_distance
            *
            reward_ratio
        )

    return (
        entry,
        stop_loss,
        take_profit,
        reward_ratio
    )


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:

    st.session_state[
        "results"
    ] = None

if "errors" not in st.session_state:

    st.session_state[
        "errors"
    ] = pd.DataFrame()


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
    index=0
)


horizon_label = st.sidebar.selectbox(
    "Primary Horizon",
    list(
        HORIZONS.keys()
    ),
    index=1,
    help=(
        "15m: "
        "5 candle = 75 menit, "
        "10 candle = 2.5 jam, "
        "20 candle = 5 jam."
    )
)


horizon = HORIZONS[
    horizon_label
]


coin_count = st.sidebar.selectbox(
    "Jumlah coin",
    [20, 40, 60, 79],
    index=3
)


min_score = st.sidebar.slider(
    "Minimum Final Trading Score",
    40,
    95,
    70,
    1
)


min_confidence = st.sidebar.slider(
    "Minimum Confidence (%)",
    40,
    95,
    60,
    1
)


min_win_rate = st.sidebar.slider(
    "Minimum Historical Win Rate (%)",
    50,
    80,
    55,
    1
)


workers = st.sidebar.slider(
    "Parallel workers",
    1,
    10,
    5,
    1
)


use_1h_confirmation = (
    st.sidebar.checkbox(
        "Gunakan konfirmasi 1H",
        value=True,
        help=(
            "Untuk scanner 15m, "
            "1H hanya menjadi "
            "filter konfirmasi tambahan."
        )
    )
)


scan_button = st.sidebar.button(
    "🚀 SCAN FUTURES",
    type="primary",
    use_container_width=True
)


clear_button = st.sidebar.button(
    "🗑️ CLEAR CACHE",
    use_container_width=True
)


if clear_button:

    st.cache_data.clear()

    st.session_state.pop(
        "results",
        None
    )

    st.session_state.pop(
        "errors",
        None
    )

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 GPR CRYPTO FUTURES SCANNER"
)

st.caption(
    f"Target: USDT Perpetual · "
    f"Timeframe {timeframe} · "
    "Ranking utama = Final Trading Score · "
    "Validasi forward 5–10 candle"
)


st.info(
    "Public OKX market data, tanpa API key. "
    "Ranking tidak dipaksa memilih coin. "
    "Jika edge historis tidak cukup kuat → NO TRADE."
)


# ============================================================
# SCAN
# ============================================================

if (
    scan_button
    or
    st.session_state[
        "results"
    ] is None
):

    selected_symbols = (
        UNIVERSE[
            :coin_count
        ]
    )

    results = []

    errors = []

    progress = st.progress(
        0
    )

    status_text = st.empty()

    start_time = time.time()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {

            executor.submit(
                analyze_symbol,
                symbol,
                timeframe,
                horizon,
                use_1h_confirmation
            ):
            symbol

            for symbol
            in selected_symbols
        }

        total = len(
            futures
        )

        for completed, future in enumerate(
            as_completed(
                futures
            ),
            start=1
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
                        "Symbol":
                            symbol,
                        "Status":
                            error
                    }
                )

            progress.progress(
                completed
                /
                total
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
                    "final_score",
                    "confidence",
                    "historical_validation"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )
            .reset_index(
                drop=True
            )
        )

        results_df[
            "rank"
        ] = np.arange(
            1,
            len(
                results_df
            ) + 1
        )

    st.session_state[
        "results"
    ] = results_df

    st.session_state[
        "errors"
    ] = errors_df

    st.session_state[
        "scan_elapsed"
    ] = (
        time.time()
        -
        start_time
    )

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
        pd.DataFrame()
    )
)

errors_df = (
    st.session_state.get(
        "errors",
        pd.DataFrame()
    )
)


if results_df.empty:

    st.warning(
        "Belum ada hasil scan. "
        "Tekan SCAN FUTURES."
    )

    st.stop()


# ============================================================
# CANDIDATES
# ============================================================

candidates = (
    results_df[
        (
            results_df[
                "final_score"
            ]
            >= min_score
        )

        &

        (
            results_df[
                "confidence"
            ]
            >= min_confidence
        )

        &

        (
            results_df[
                "historical_validation"
            ]
            * 100
            >= min_win_rate
        )

        &

        (
            results_df[
                "trade_status"
            ]
            .isin(
                [
                    "TRADEABLE",
                    "WATCH"
                ]
            )
        )
    ]
    .copy()
)


tradeable = candidates[
    candidates[
        "trade_status"
    ]
    ==
    "TRADEABLE"
].copy()


# ============================================================
# SUMMARY
# ============================================================

col1, col2, col3, col4, col5 = (
    st.columns(5)
)


col1.metric(
    "UNIVERSE",
    len(
        results_df
    ),
    "USDT Perpetual"
)


col2.metric(
    "TIMEFRAME",
    timeframe,
    "Primary"
)


hours = (
    horizon
    *
    (
        0.25
        if timeframe == "15m"
        else 1
    )
)


col3.metric(
    "HORIZON",
    f"{horizon} candle",
    f"≈ {hours:g} jam"
)


col4.metric(
    "TRADEABLE",
    len(
        tradeable
    )
)


best = results_df.iloc[0]


col5.metric(
    "TOP FINAL SCORE",
    f"{best['final_score']:.1f}",
    best["symbol"]
)


st.success(
    f"Scan selesai: "
    f"{len(results_df)} berhasil, "
    f"{len(errors_df)} gagal "
    f"dalam "
    f"{st.session_state.get('scan_elapsed', 0):.1f} detik."
)


# ============================================================
# RANKING
# ============================================================

st.divider()

st.subheader(
    "🏆 Trading Ranking"
)


display = results_df[
    [
        "rank",
        "symbol",
        "signal",
        "trade_status",
        "final_score",
        "gpr_score",
        "forecast",
        "uncertainty",
        "confidence",
        "confidence_ratio",
        "historical_validation",
        "recent_validation",
        "validation_samples",
        "confirmation_1h",
        "price",
        "rsi",
        "mom6",
        "mom12",
        "mom24",
        "vol_z"
    ]
].copy()


display.columns = [
    "Rank",
    "Coin",
    "Signal",
    "Status",
    "Final Score",
    "GPR Score",
    "Forecast",
    "Uncertainty",
    "Confidence",
    "Conf Ratio",
    "Hist Win Rate",
    "Recent Win Rate",
    "Samples",
    "1H",
    "Price",
    "RSI",
    "Mom 6H",
    "Mom 12H",
    "Mom 24H",
    "Vol Z"
]


display[
    "Final Score"
] = display[
    "Final Score"
].map(
    lambda x:
        f"{x:.1f}"
)


display[
    "GPR Score"
] = display[
    "GPR Score"
].map(
    lambda x:
        f"{x:.1f}"
)


display[
    "Forecast"
] = display[
    "Forecast"
].map(
    lambda x:
        f"{x:+.2f}%"
)


display[
    "Uncertainty"
] = display[
    "Uncertainty"
].map(
    lambda x:
        f"{x:.2f}%"
)


display[
    "Confidence"
] = display[
    "Confidence"
].map(
    lambda x:
        f"{x:.0f}%"
)


display[
    "Conf Ratio"
] = display[
    "Conf Ratio"
].map(
    lambda x:
        f"{x:.2f}x"
)


display[
    "Hist Win Rate"
] = display[
    "Hist Win Rate"
].map(
    lambda x:
        f"{x * 100:.0f}%"
)


display[
    "Recent Win Rate"
] = display[
    "Recent Win Rate"
].map(
    lambda x:
        f"{x * 100:.0f}%"
)


display[
    "Price"
] = display[
    "Price"
].map(
    fmt_price
)


display[
    "RSI"
] = display[
    "RSI"
].map(
    lambda x:
        f"{x:.1f}"
)


for column in [
    "Mom 6H",
    "Mom 12H",
    "Mom 24H"
]:

    display[
        column
    ] = display[
        column
    ].map(
        lambda x:
            f"{x:+.2f}%"
    )


display[
    "Vol Z"
] = display[
    "Vol Z"
].map(
    lambda x:
        f"{x:+.2f}"
)


st.dataframe(
    display,
    use_container_width=True,
    hide_index=True
)


st.caption(
    "Ranking utama memakai Final Trading Score. "
    "GPR Score tetap ditampilkan sebagai indikator kualitas setup. "
    "Historical Win Rate adalah hasil walk-forward pada candle historis."
)


# ============================================================
# TOP TRADEABLE SIGNALS
# ============================================================

st.divider()

st.subheader(
    "🎯 Top Tradeable Signals"
)


top = tradeable.head(6)


if top.empty:

    st.warning(
        "Tidak ada coin yang memenuhi semua filter. "
        "Ini disengaja: scanner tidak memaksa menghasilkan "
        "trade jika edge 5–10 candle tidak cukup kuat."
    )

else:

    cards = st.columns(2)

    for i, (
        _,
        row
    ) in enumerate(
        top.iterrows()
    ):

        with cards[
            i % 2
        ]:

            direction = (
                "🟢 LONG"
                if row["signal"]
                ==
                "LONG"
                else
                "🔴 SHORT"
            )

            st.markdown(
                f"### "
                f"{row['symbol']} · "
                f"{direction}"
            )

            a, b = (
                st.columns(2)
            )

            a.metric(
                "Final Score",
                f"{row['final_score']:.1f}"
            )

            b.metric(
                "Confidence",
                f"{row['confidence']:.0f}%"
            )

            a, b = (
                st.columns(2)
            )

            a.metric(
                f"Forecast {horizon}C",
                pct(
                    row["forecast"]
                )
            )

            b.metric(
                "Hist Win Rate",
                f"{row['historical_validation'] * 100:.0f}%"
            )

            st.caption(
                f"Recent WR "
                f"{row['recent_validation'] * 100:.0f}% · "
                f"Agreement "
                f"{row['agreement']:.0f}% · "
                f"Conf Ratio "
                f"{row['confidence_ratio']:.2f}x · "
                f"1H "
                f"{row['confirmation_1h']} · "
                f"{int(row['validation_samples'])} samples"
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
    ]
    .tolist()
)


if not detail_symbols:

    detail_symbols = (
        results_df[
            "symbol"
        ]
        .tolist()
    )


selected_symbol = st.selectbox(
    "Pilih coin",
    detail_symbols
)


selected = (
    results_df[
        results_df[
            "symbol"
        ]
        ==
        selected_symbol
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
    selected[
        "signal"
    ]
)


b.metric(
    "FINAL SCORE",
    f"{selected['final_score']:.1f}"
)


c.metric(
    "FORECAST",
    pct(
        selected[
            "forecast"
        ]
    )
)


d.metric(
    "STATUS",
    selected[
        "trade_status"
    ]
)


# ============================================================
# CONFIDENCE
# ============================================================

st.markdown(
    "#### 📊 Confidence & Validation"
)


confidence = float(
    selected[
        "confidence"
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
    f"Confidence "
    f"{confidence:.0f}%**. "

    "Confidence adalah indeks "
    "kekuatan dukungan model, "
    "bukan probabilitas pasti profit. "

    f"Confidence Ratio = "
    f"{selected['confidence_ratio']:.2f}x."
)


a, b, c, d = (
    st.columns(4)
)


a.metric(
    "Confidence",
    f"{confidence:.0f}%"
)


b.metric(
    "Confidence Ratio",
    f"{selected['confidence_ratio']:.2f}x"
)


c.metric(
    "Signal Agreement",
    f"{selected['agreement']:.0f}%"
)


d.metric(
    "Historical Win Rate",
    f"{selected['historical_validation'] * 100:.0f}%"
)


a, b, c, d = (
    st.columns(4)
)


a.metric(
    "Recent Win Rate",
    f"{selected['recent_validation'] * 100:.0f}%"
)


b.metric(
    "Validation Samples",
    int(
        selected[
            "validation_samples"
        ]
    )
)


c.metric(
    "Hist Mean Return",
    pct(
        selected[
            "historical_mean_return"
        ]
    )
)


d.metric(
    "Hist Median Return",
    pct(
        selected[
            "historical_median_return"
        ]
    )
)


st.progress(
    int(
        np.clip(
            confidence,
            0,
            100
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
        selected[
            "price"
        ]
    )
)


b.metric(
    "RSI 14",
    f"{selected['rsi']:.1f}"
)


c.metric(
    "Momentum 6H",
    pct(
        selected[
            "mom6"
        ]
    )
)


d.metric(
    "Momentum 12H",
    pct(
        selected[
            "mom12"
        ]
    )
)


a, b, c, d = (
    st.columns(4)
)


a.metric(
    "Momentum 24H",
    pct(
        selected[
            "mom24"
        ]
    )
)


b.metric(
    "Volume Z",
    f"{selected['vol_z']:+.2f}"
)


c.metric(
    "EMA12 Distance",
    pct(
        selected[
            "ema12_dist"
        ]
    )
)


d.metric(
    "EMA24 Distance",
    pct(
        selected[
            "ema24_dist"
        ]
    )
)


last_candle = datetime.fromtimestamp(
    selected[
        "timestamp"
    ] / 1000,
    tz=timezone.utc
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
    1.0,
    3.0,
    1.67,
    0.01
)


entry, stop_loss, take_profit, rr = (
    calculate_risk_plan(
        selected,
        reward_ratio
    )
)


a, b, c, d = (
    st.columns(4)
)


a.metric(
    "Entry / Current",
    fmt_price(entry)
)


b.metric(
    "Stop Loss",
    fmt_price(stop_loss)
)


c.metric(
    "Take Profit",
    fmt_price(take_profit)
)


d.metric(
    "Risk / Reward",
    f"1 : {rr:.2f}"
)


st.caption(
    "Stop Loss menggunakan uncertainty model. "
    "Take Profit mengikuti target Risk/Reward. "
    "Ini alat bantu analisis, bukan instruksi trading otomatis."
)


# ============================================================
# SIGNAL COMPONENTS
# ============================================================

st.divider()

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
            "Recent validation",
            "Signal agreement",
            "1H confirmation",
        ],

        "Value": [
            selected[
                "momentum_component"
            ],

            selected[
                "trend_component"
            ],

            selected[
                "rsi_component"
            ],

            selected[
                "volume_component"
            ],

            selected[
                "historical_validation"
            ],

            selected[
                "recent_validation"
            ],

            selected[
                "agreement"
            ] / 100,

            selected[
                "confirmation_1h"
            ],
        ],
    }
)


st.dataframe(
    components,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown(
    "#### 📝 Interpretasi"
)


signal = selected[
    "signal"
]

status = selected[
    "trade_status"
]


st.write(
    f"**{selected_symbol}**: "
    f"**{signal}**, "
    f"Final Trading Score "
    f"**{selected['final_score']:.1f}**, "
    f"GPR Score "
    f"**{selected['gpr_score']:.1f}**, "
    f"Confidence "
    f"**{selected['confidence']:.0f}%**. "

    f"Forecast "
    f"{horizon} candle = "
    f"**{pct(selected['forecast'])}**. "

    f"Historical Win Rate = "
    f"**{selected['historical_validation'] * 100:.0f}%** "
    f"("
    f"{int(selected['validation_samples'])}"
    f" sample). "

    f"Status: **{status}**."
)


# ============================================================
# 1H INTERPRETATION
# ============================================================

if selected[
    "confirmation_1h"
] != "UNKNOWN":

    if (
        signal == "LONG"
        and
        selected[
            "confirmation_1h"
        ]
        ==
        "BULLISH"
    ):

        st.success(
            "Konfirmasi 1H searah dengan LONG."
        )

    elif (
        signal == "SHORT"
        and
        selected[
            "confirmation_1h"
        ]
        ==
        "BEARISH"
    ):

        st.success(
            "Konfirmasi 1H searah dengan SHORT."
        )

    elif selected[
        "confirmation_1h"
    ] == "NEUTRAL":

        st.warning(
            "Konfirmasi 1H masih NEUTRAL."
        )

    else:

        st.warning(
            f"Signal {signal} "
            f"belum searah dengan "
            f"konfirmasi 1H "
            f"({selected['confirmation_1h']})."
        )


# ============================================================
# DIAGNOSTIC
# ============================================================

st.divider()

with st.expander(
    f"📋 ERROR / DIAGNOSTIC "
    f"({len(errors_df)})"
):

    if errors_df.empty:

        st.success(
            "Tidak ada error."
        )

    else:

        st.dataframe(
            errors_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# EXPORT
# ============================================================

csv = (
    results_df
    .to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)


st.download_button(
    "⬇️ Download Ranking CSV",
    data=csv,
    file_name=(
        f"gpr_crypto_"
        f"{timeframe}_"
        f"{horizon}c.csv"
    ),
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GPR Crypto Futures Scanner · "
    "OKX Public Market Data · "
    "15m / 1H · "
    "Research Prototype · "
    "Tidak melakukan order otomatis."
)
