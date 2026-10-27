import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler


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
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2.2rem !important;
    }

    .small-text {
        color: #6b7280;
        font-size: 0.85rem;
    }

    .long-text {
        color: #16a34a;
        font-weight: 700;
    }

    .short-text {
        color: #dc2626;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANT
# ============================================================

OKX_BASE_URL = "https://www.okx.com"

TIMEFRAMES = {
    "15m": {
        "bar": "15m",
        "minutes": 15,
        "label": "15 menit",
    },
    "1H": {
        "bar": "1H",
        "minutes": 60,
        "label": "1 jam",
    },
}


# ============================================================
# STATIC FUTURES UNIVERSE
# ============================================================
# Sengaja menggunakan daftar static.
# Tidak bergantung pada endpoint instruments/exchangeInfo
# yang sebelumnya menghasilkan HTTP 403/451.

UNIVERSE = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "BNB",
    "ADA",
    "AVAX",
    "LINK",
    "DOT",
    "TRX",
    "SUI",
    "TON",
    "SHIB",
    "LTC",
    "BCH",
    "NEAR",
    "APT",
    "ICP",
    "FIL",
    "ATOM",
    "ETC",
    "ARB",
    "OP",
    "INJ",
    "IMX",
    "AAVE",
    "MKR",
    "RUNE",
    "UNI",
    "LDO",
    "STX",
    "SEI",
    "TIA",
    "JUP",
    "WIF",
    "PEPE",
    "BONK",
    "FLOKI",
    "WLD",
    "ENA",
    "ONDO",
    "PYTH",
    "JTO",
    "STRK",
    "NOT",
    "ORDI",
    "SATS",
    "SAND",
    "MANA",
    "AXS",
    "GALA",
    "EOS",
    "XLM",
    "VET",
    "ALGO",
    "HBAR",
    "FTM",
    "APE",
    "CRV",
    "COMP",
    "SNX",
    "DYDX",
    "1INCH",
    "KAS",
    "TAO",
    "FET",
    "RENDER",
    "MANTA",
    "DYM",
    "WOO",
    "JASMY",
    "KAVA",
    "MINA",
    "CELO",
    "ZRX",
    "QTUM",
]


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "rsi14",
    "mom_6",
    "mom_12",
    "mom_24",
    "volume_z",
    "ema12_dist",
    "ema24_dist",
    "ema_spread",
    "atr_pct",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def bars_for_hours(hours, timeframe):
    minutes = TIMEFRAMES[timeframe]["minutes"]
    return max(1, int(round(hours * 60 / minutes)))


def horizon_label(horizon_bars, timeframe):
    minutes = (
        horizon_bars
        * TIMEFRAMES[timeframe]["minutes"]
    )

    if minutes >= 60:
        hours = minutes / 60
        return f"{horizon_bars} candle / {hours:g} jam"

    return f"{horizon_bars} candle / {minutes:g} menit"


# ============================================================
# CONFIDENCE
# ============================================================

def normal_cdf(x):
    return 0.5 * (
        1.0 + math.erf(x / math.sqrt(2.0))
    )


def get_confidence(forecast, uncertainty):
    """
    Confidence ratio:

        abs(forecast) / uncertainty

    Ini BUKAN probabilitas profit.

    Contoh:
        0.5  = sangat lemah
        1.0  = lemah
        1.5  = sedang
        2.0  = baik
        2.5  = kuat
        3.0+ = sangat kuat
    """

    uncertainty = max(
        abs(float(uncertainty)),
        0.000001,
    )

    forecast = float(forecast)

    ratio = abs(forecast) / uncertainty

    probability = np.clip(
        normal_cdf(ratio),
        0.50,
        0.9999,
    )

    if ratio < 0.50:
        label = "SANGAT LEMAH"
    elif ratio < 1.00:
        label = "LEMAH"
    elif ratio < 1.50:
        label = "SEDANG"
    elif ratio < 2.00:
        label = "BAIK"
    elif ratio < 2.50:
        label = "KUAT"
    elif ratio < 3.00:
        label = "SANGAT KUAT"
    else:
        label = "EKSTREM"

    return (
        float(ratio),
        float(probability * 100),
        label,
    )


# ============================================================
# OKX MARKET DATA
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def get_okx_candles(symbol, timeframe, limit=300):

    bar = TIMEFRAMES[timeframe]["bar"]

    inst_id = f"{symbol}-USDT-SWAP"

    url = (
        f"{OKX_BASE_URL}"
        "/api/v5/market/candles"
    )

    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": str(limit),
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/130 Safari/537.36"
        )
    }

    last_error = "Unknown error"

    for attempt in range(3):

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=(5, 20),
            )

            response.raise_for_status()

            data = response.json()

            code = str(
                data.get("code", "")
            )

            if code != "0":

                message = data.get(
                    "msg",
                    "OKX API error",
                )

                raise RuntimeError(
                    f"OKX code={code}: {message}"
                )

            rows = data.get("data", [])

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

            df["datetime"] = pd.to_datetime(
                df["ts"],
                unit="ms",
                utc=True,
                errors="coerce",
            )

            # Hanya candle confirmed.
            if "confirm" in df.columns:

                df = df[
                    df["confirm"]
                    .astype(str)
                    .eq("1")
                ]

            df = (
                df
                .sort_values("datetime")
                .dropna(
                    subset=[
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "datetime",
                    ]
                )
                .reset_index(drop=True)
            )

            if len(df) < 120:

                raise RuntimeError(
                    f"Data candle kurang: "
                    f"{len(df)} bar."
                )

            return df

        except Exception as error:

            last_error = str(error)

            if attempt < 2:
                time.sleep(
                    0.7 * (attempt + 1)
                )

    raise RuntimeError(last_error)


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    series,
    period=14,
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
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    return (
        100
        - (
            100
            / (1 + rs)
        )
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    df,
    timeframe,
):

    x = df.copy()

    close = x["close"]
    volume = x["volume"]

    b6 = bars_for_hours(
        6,
        timeframe,
    )

    b12 = bars_for_hours(
        12,
        timeframe,
    )

    b24 = bars_for_hours(
        24,
        timeframe,
    )

    # RSI
    x["rsi14"] = calculate_rsi(
        close,
        14,
    )

    # Momentum
    x["mom_6"] = (
        close
        / close.shift(b6)
        - 1
    ) * 100

    x["mom_12"] = (
        close
        / close.shift(b12)
        - 1
    ) * 100

    x["mom_24"] = (
        close
        / close.shift(b24)
        - 1
    ) * 100

    # EMA
    x["ema12"] = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    x["ema24"] = close.ewm(
        span=24,
        adjust=False,
    ).mean()

    x["ema12_dist"] = (
        close
        / x["ema12"]
        - 1
    ) * 100

    x["ema24_dist"] = (
        close
        / x["ema24"]
        - 1
    ) * 100

    x["ema_spread"] = (
        x["ema12"]
        / x["ema24"]
        - 1
    ) * 100

    # Volume Z-score
    volume_mean = (
        volume
        .rolling(30)
        .mean()
    )

    volume_std = (
        volume
        .rolling(30)
        .std()
    )

    x["volume_z"] = (
        volume
        - volume_mean
    ) / volume_std.replace(
        0,
        np.nan,
    )

    # ATR %
    previous_close = close.shift(1)

    tr1 = (
        x["high"]
        - x["low"]
    )

    tr2 = (
        x["high"]
        - previous_close
    ).abs()

    tr3 = (
        x["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr14 = (
        true_range
        .rolling(14)
        .mean()
    )

    x["atr_pct"] = (
        atr14
        / close
    ) * 100

    return x


# ============================================================
# GPR MODEL
# ============================================================

def train_gpr(
    candles,
    timeframe,
    horizon_bars,
):

    data = build_features(
        candles,
        timeframe,
    )

    # Target = return beberapa candle ke depan.
    data["future_return"] = (
        data["close"].shift(
            -horizon_bars
        )
        / data["close"]
        - 1
    ) * 100

    model_data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=FEATURES
            + ["future_return"]
        )
        .tail(180)
        .copy()
    )

    if len(model_data) < 70:

        raise RuntimeError(
            "Training data tidak cukup."
        )

    X = model_data[FEATURES]
    y = model_data[
        "future_return"
    ]

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 70:

        raise RuntimeError(
            "Training rows valid "
            "kurang dari 70."
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    kernel = (
        ConstantKernel(
            1.0,
            (
                1e-3,
                1e3,
            ),
        )
        * RBF(
            1.0,
            length_scale_bounds=(
                1e-2,
                1e2,
            ),
        )
        + WhiteKernel(
            0.5,
            noise_level_bounds=(
                1e-5,
                10,
            ),
        )
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=42,
    )

    model.fit(
        X_scaled,
        y.to_numpy(),
    )

    latest = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=FEATURES
        )
        .iloc[-1]
    )

    latest_x = (
        latest[FEATURES]
        .to_numpy(
            dtype=float
        )
        .reshape(
            1,
            -1,
        )
    )

    prediction, std = (
        model.predict(
            scaler.transform(
                latest_x
            ),
            return_std=True,
        )
    )

    forecast = float(
        prediction[0]
    )

    uncertainty = max(
        float(std[0]),
        0.000001,
    )

    return (
        forecast,
        uncertainty,
        latest,
    )


# ============================================================
# GPR SCORE
# ============================================================

def calculate_gpr_score(
    forecast,
    uncertainty,
    latest,
):

    confidence, confidence_pct, confidence_label = (
        get_confidence(
            forecast,
            uncertainty,
        )
    )

    # --------------------------------------------------------
    # 1. Magnitude
    # --------------------------------------------------------

    magnitude = np.clip(
        abs(forecast) / 5.0,
        0,
        1,
    )

    # --------------------------------------------------------
    # 2. Confidence
    # --------------------------------------------------------

    confidence_component = np.clip(
        (confidence - 0.5) / 2.5,
        0,
        1,
    )

    # --------------------------------------------------------
    # 3. RSI
    # --------------------------------------------------------

    rsi = safe_float(
        latest["rsi14"],
        50,
    )

    if forecast >= 0:

        if 45 <= rsi <= 70:
            rsi_component = 1.00

        elif 35 <= rsi < 45:
            rsi_component = 0.75

        elif 70 < rsi <= 78:
            rsi_component = 0.75

        else:
            rsi_component = 0.40

    else:

        if 30 <= rsi <= 55:
            rsi_component = 1.00

        elif 22 <= rsi < 30:
            rsi_component = 0.75

        elif 55 < rsi <= 65:
            rsi_component = 0.75

        else:
            rsi_component = 0.40

    # --------------------------------------------------------
    # 4. Momentum
    # --------------------------------------------------------

    momentum = np.nanmean(
        [
            safe_float(
                latest["mom_6"],
                0,
            ),
            safe_float(
                latest["mom_12"],
                0,
            ),
            safe_float(
                latest["mom_24"],
                0,
            ),
        ]
    )

    if forecast >= 0:

        if momentum > 0.10:
            momentum_component = 1.00

        elif momentum >= 0:
            momentum_component = 0.75

        else:
            momentum_component = 0.35

    else:

        if momentum < -0.10:
            momentum_component = 1.00

        elif momentum <= 0:
            momentum_component = 0.75

        else:
            momentum_component = 0.35

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = (
        magnitude * 35
        + confidence_component * 30
        + rsi_component * 15
        + momentum_component * 20
    )

    return (
        float(
            np.clip(
                score,
                0,
                100,
            )
        ),
        confidence,
        confidence_pct,
        confidence_label,
    )


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_symbol(
    symbol,
    timeframe,
    horizon_bars,
):

    candles = get_okx_candles(
        symbol,
        timeframe,
        300,
    )

    forecast, uncertainty, latest = (
        train_gpr(
            candles,
            timeframe,
            horizon_bars,
        )
    )

    (
        score,
        confidence,
        confidence_pct,
        confidence_label,
    ) = calculate_gpr_score(
        forecast,
        uncertainty,
        latest,
    )

    signal = (
        "LONG"
        if forecast >= 0
        else "SHORT"
    )

    return {
        "COIN": symbol,
        "SIGNAL": signal,
        "GPR SCORE": score,
        "GPR RETURN": forecast,
        "UNCERTAINTY": uncertainty,
        "CONFIDENCE": confidence,
        "CONF %": confidence_pct,
        "CONF LABEL": confidence_label,
        "PRICE": float(
            latest["close"]
        ),
        "RSI": float(
            latest["rsi14"]
        ),
        "MOM 6H": float(
            latest["mom_6"]
        ),
        "MOM 12H": float(
            latest["mom_12"]
        ),
        "MOM 24H": float(
            latest["mom_24"]
        ),
        "VOL Z": float(
            latest["volume_z"]
        ),
        "EMA12 DIST": float(
            latest["ema12_dist"]
        ),
        "EMA24 DIST": float(
            latest["ema24_dist"]
        ),
        "CANDLE": latest[
            "datetime"
        ],
    }


# ============================================================
# SCAN
# ============================================================

def run_scan(
    symbols,
    timeframe,
    horizon_bars,
    workers,
):

    results = []
    errors = []

    progress = st.progress(
        0
    )

    status = st.empty()

    total = len(symbols)

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                analyze_symbol,
                symbol,
                timeframe,
                horizon_bars,
            ): symbol
            for symbol in symbols
        }

        completed = 0

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            completed += 1

            status.write(
                f"Scanning {symbol} "
                f"({completed}/{total})..."
            )

            try:

                result = (
                    future.result()
                )

                results.append(
                    result
                )

            except Exception as error:

                errors.append(
                    {
                        "Symbol": symbol,
                        "Status": str(
                            error
                        ),
                    }
                )

            progress.progress(
                completed / total
            )

    progress.empty()
    status.empty()

    return (
        results,
        errors,
    )


# ============================================================
# FORMAT TABLE
# ============================================================

def make_display_table(
    dataframe,
):

    columns = [
        "Rank",
        "COIN",
        "SIGNAL",
        "GPR SCORE",
        "GPR RETURN",
        "UNCERTAINTY",
        "CONFIDENCE",
        "CONF %",
        "CONF LABEL",
        "PRICE",
        "RSI",
        "MOM 6H",
        "MOM 12H",
        "MOM 24H",
        "VOL Z",
    ]

    output = dataframe[
        columns
    ].copy()

    output[
        "GPR SCORE"
    ] = output[
        "GPR SCORE"
    ].round(1)

    output[
        "GPR RETURN"
    ] = output[
        "GPR RETURN"
    ].map(
        lambda x: f"{x:+.2f}%"
    )

    output[
        "UNCERTAINTY"
    ] = output[
        "UNCERTAINTY"
    ].map(
        lambda x: f"{x:.2f}%"
    )

    output[
        "CONFIDENCE"
    ] = output[
        "CONFIDENCE"
    ].map(
        lambda x: f"{x:.2f}"
    )

    output[
        "CONF %"
    ] = output[
        "CONF %"
    ].map(
        lambda x: f"{x:.1f}%"
    )

    output[
        "PRICE"
    ] = output[
        "PRICE"
    ].map(
        lambda x: f"{x:.8g}"
    )

    output[
        "RSI"
    ] = output[
        "RSI"
    ].round(1)

    for column in [
        "MOM 6H",
        "MOM 12H",
        "MOM 24H",
    ]:

        output[
            column
        ] = output[
            column
        ].map(
            lambda x: f"{x:+.2f}%"
        )

    output[
        "VOL Z"
    ] = output[
        "VOL Z"
    ].round(2)

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "⚙️ Scanner Settings"
    )

    timeframe = st.selectbox(
        "Timeframe",
        options=[
            "15m",
            "1H",
        ],
        index=0,
    )

    horizon_bars = st.selectbox(
        "Horizon GPR",
        options=[
            4,
            6,
            8,
            10,
            12,
            16,
            20,
        ],
        index=3,
        format_func=lambda x: horizon_label(
            x,
            timeframe,
        ),
    )

    coin_count = st.selectbox(
        "Jumlah coin",
        options=[
            20,
            30,
            50,
            79,
        ],
        index=3,
    )

    minimum_score = st.slider(
        "Minimum GPR Score",
        min_value=40,
        max_value=95,
        value=70,
        step=1,
    )

    minimum_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
    )

    workers = st.slider(
        "Parallel workers",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    st.divider()

    scan_button = st.button(
        "🚀 SCAN FUTURES",
        type="primary",
        use_container_width=True,
    )

    clear_button = st.button(
        "🗑️ CLEAR CACHE",
        use_container_width=True,
    )

    if clear_button:

        st.cache_data.clear()

        st.success(
            "Cache berhasil dibersihkan."
        )

        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if "scan_errors" not in st.session_state:
    st.session_state.scan_errors = []

if "scan_time" not in st.session_state:
    st.session_state.scan_time = 0.0


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 GPR CRYPTO FUTURES SCANNER"
)

st.caption(
    "Target utama: USDT Perpetual · "
    f"Timeframe {timeframe} · "
    "GPR sebagai mesin ranking utama"
)


st.info(
    "Mode data: public market data, tanpa API key. "
    "Universe futures menggunakan daftar static "
    "agar scanner tidak bergantung pada endpoint "
    "instruments/exchangeInfo yang sebelumnya "
    "menghasilkan HTTP 403/451."
)


# ============================================================
# SCAN
# ============================================================

if scan_button:

    selected_symbols = (
        UNIVERSE[:coin_count]
    )

    start_time = time.time()

    results, errors = run_scan(
        selected_symbols,
        timeframe,
        horizon_bars,
        workers,
    )

    elapsed = (
        time.time()
        - start_time
    )

    st.session_state.scan_results = (
        results
    )

    st.session_state.scan_errors = (
        errors
    )

    st.session_state.scan_time = (
        elapsed
    )


# ============================================================
# GET RESULTS
# ============================================================

results = (
    st.session_state.scan_results
)

errors = (
    st.session_state.scan_errors
)


if results is None:

    st.info(
        "Tekan **SCAN FUTURES** untuk "
        "memulai analisis."
    )

else:

    df = pd.DataFrame(
        results
    )

    if df.empty:

        st.error(
            "Tidak ada coin yang berhasil "
            "dianalisis."
        )

        if errors:

            error_df = pd.DataFrame(
                errors
            )

            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True,
            )

        st.stop()


    # ========================================================
    # SORT
    # ========================================================

    df = df.sort_values(
        [
            "GPR SCORE",
            "CONFIDENCE",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    df["Rank"] = (
        df.index + 1
    )


    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = df[
        (
            df["GPR SCORE"]
            >= minimum_score
        )
        &
        (
            df["CONFIDENCE"]
            >= minimum_confidence
        )
    ].copy()


    # ========================================================
    # TOP RESULT
    # ========================================================

    top_row = df.iloc[0]

    top_return = (
        top_row["GPR RETURN"]
    )

    top_score = (
        top_row["GPR SCORE"]
    )


    # ========================================================
    # STATUS
    # ========================================================

    st.success(
        f"Scan selesai: "
        f"{len(results)} berhasil, "
        f"{len(errors)} gagal "
        f"dalam "
        f"{st.session_state.scan_time:.1f} detik."
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "UNIVERSE",
            len(results),
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
            f"{horizon_bars} candle",
            horizon_label(
                horizon_bars,
                timeframe,
            ),
        )

    with c4:

        st.metric(
            "PROVIDER",
            "OKX",
            "Public market data",
        )


    # ========================================================
    # SECOND METRICS
    # ========================================================

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "BERHASIL DIANALISIS",
            len(results),
        )

    with c2:

        st.metric(
            f"LOLOS SCORE ≥ {minimum_score}",
            int(
                (
                    df["GPR SCORE"]
                    >= minimum_score
                ).sum()
            ),
        )

    with c3:

        st.metric(
            "KANDIDAT",
            len(candidates),
        )

    with c4:

        st.metric(
            "GPR RETURN TERTINGGI",
            f"{top_return:+.2f}%",
            f"Score {top_score:.1f}",
        )


    # ========================================================
    # RANKING
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 GPR Futures Ranking"
    )

    display_df = (
        make_display_table(df)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=520,
    )


    # ========================================================
    # CONFIDENCE GUIDE
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Cara Membaca Confidence"
    )

    guide = pd.DataFrame(
        {
            "Confidence": [
                "< 0.50",
                "0.50 – 0.99",
                "1.00 – 1.49",
                "1.50 – 1.99",
                "2.00 – 2.49",
                "2.50 – 2.99",
                "≥ 3.00",
            ],
            "Level": [
                "SANGAT LEMAH",
                "LEMAH",
                "SEDANG",
                "BAIK",
                "KUAT",
                "SANGAT KUAT",
                "EKSTREM",
            ],
            "Interpretasi": [
                "Signal sangat noisy",
                "Signal lemah",
                "Mulai terlihat",
                "Cukup meyakinkan",
                "Signal kuat",
                "Signal sangat kuat",
                "Forecast jauh di atas uncertainty",
            ],
        }
    )

    st.dataframe(
        guide,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Penting: Confidence bukan probabilitas "
        "bahwa trade pasti profit. Confidence adalah "
        "rasio |GPR Return| terhadap uncertainty. "
        "Semakin tinggi, semakin besar forecast "
        "dibandingkan ketidakpastian model."
    )


    # ========================================================
    # TOP GPR SIGNALS
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Top GPR Signals"
    )

    top_signals = (
        df
        .sort_values(
            [
                "GPR SCORE",
                "CONFIDENCE",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(6)
    )

    signal_columns = st.columns(2)

    for index, (_, row) in enumerate(
        top_signals.iterrows()
    ):

        column = signal_columns[
            index % 2
        ]

        with column:

            signal = row[
                "SIGNAL"
            ]

            signal_icon = (
                "🟢"
                if signal == "LONG"
                else "🔴"
            )

            st.markdown(
                f"### {row['COIN']}"
            )

            st.write(
                f"{signal_icon} "
                f"**{signal}**"
            )

            m1, m2 = st.columns(2)

            with m1:

                st.metric(
                    "GPR Score",
                    f"{row['GPR SCORE']:.1f}",
                )

                st.metric(
                    "GPR Return",
                    f"{row['GPR RETURN']:+.2f}%",
                )

            with m2:

                st.metric(
                    "Confidence",
                    f"{row['CONFIDENCE']:.2f}",
                )

                st.metric(
                    "Uncertainty",
                    f"{row['UNCERTAINTY']:.2f}%",
                )

            st.caption(
                f"Confidence level: "
                f"{row['CONF LABEL']}"
            )


    # ========================================================
    # DETAIL ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 Detail Analysis"
    )

    coin_list = (
        df["COIN"]
        .tolist()
    )

    selected_coin = st.selectbox(
        "Pilih coin",
        coin_list,
    )

    selected = df[
        df["COIN"]
        == selected_coin
    ].iloc[0]


    # --------------------------------------------------------
    # Main detail
    # --------------------------------------------------------

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "SIGNAL",
            selected["SIGNAL"],
        )

    with d2:

        st.metric(
            "GPR SCORE",
            f"{selected['GPR SCORE']:.1f}",
        )

    with d3:

        st.metric(
            "FORECAST",
            f"{selected['GPR RETURN']:+.2f}%",
        )

    with d4:

        st.metric(
            "UNCERTAINTY",
            f"{selected['UNCERTAINTY']:.2f}%",
        )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    st.write("### Confidence")

    confidence_col1, confidence_col2 = st.columns(2)

    with confidence_col1:

        st.metric(
            "Confidence Ratio",
            f"{selected['CONFIDENCE']:.2f}",
        )

    with confidence_col2:

        st.metric(
            "Confidence Estimate",
            f"{selected['CONF %']:.1f}%",
        )

    st.info(
        f"Level: **{selected['CONF LABEL']}**. "
        f"Nilai {selected['CONFIDENCE']:.2f} berarti "
        f"besar forecast sekitar "
        f"{selected['CONFIDENCE']:.2f}× "
        f"uncertainty model."
    )


    # --------------------------------------------------------
    # Technical data
    # --------------------------------------------------------

    st.write(
        "### Technical Data"
    )

    t1, t2, t3, t4 = st.columns(4)

    with t1:

        st.metric(
            "Harga",
            f"{selected['PRICE']:.8g}",
        )

        st.metric(
            "RSI 14",
            f"{selected['RSI']:.1f}",
        )

    with t2:

        st.metric(
            "Momentum 6H",
            f"{selected['MOM 6H']:+.2f}%",
        )

        st.metric(
            "Momentum 12H",
            f"{selected['MOM 12H']:+.2f}%",
        )

    with t3:

        st.metric(
            "Momentum 24H",
            f"{selected['MOM 24H']:+.2f}%",
        )

        st.metric(
            "Volume Z",
            f"{selected['VOL Z']:.2f}",
        )

    with t4:

        st.metric(
            "EMA12 Distance",
            f"{selected['EMA12 DIST']:+.2f}%",
        )

        st.metric(
            "EMA24 Distance",
            f"{selected['EMA24 DIST']:+.2f}%",
        )


    st.caption(
        "Last confirmed candle: "
        f"{selected['CANDLE']}"
    )


    # ========================================================
    # CANDIDATE FILTER
    # ========================================================

    st.divider()

    st.subheader(
        "⭐ Kandidat Trading"
    )

    if candidates.empty:

        st.warning(
            f"Belum ada kandidat yang memenuhi "
            f"Score ≥ {minimum_score} dan "
            f"Confidence ≥ {minimum_confidence:.1f}."
        )

        st.write(
            "Coba turunkan Minimum GPR Score "
            "atau Minimum Confidence."
        )

    else:

        candidate_display = (
            make_display_table(
                candidates
            )
        )

        st.dataframe(
            candidate_display,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    st.divider()

    with st.expander(
        f"📋 ERROR / DIAGNOSTIC ({len(errors)})"
    ):

        if not errors:

            st.success(
                "Tidak ada error."
            )

        else:

            error_df = pd.DataFrame(
                errors
            )

            st.dataframe(
                error_df,
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # DOWNLOAD CSV
    # ========================================================

    st.divider()

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Ranking CSV",
        data=csv_data,
        file_name=(
            f"gpr_crypto_"
            f"{timeframe}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GPR Crypto Futures Scanner · "
    "USDT Perpetual · "
    "OKX Public Market Data · "
    "Research Prototype"
)
