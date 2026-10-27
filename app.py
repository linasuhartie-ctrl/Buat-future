import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import math

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GPR Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 2.4rem !important;
    margin-bottom: 0.2rem !important;
}

.small-muted {
    color: #777;
    font-size: 0.9rem;
}

.signal-long {
    color: #00a65a;
    font-weight: 800;
}

.signal-short {
    color: #e53935;
    font-weight: 800;
}

.metric-box {
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    background: #fafafa;
    min-height: 110px;
}

.metric-title {
    font-size: 0.82rem;
    color: #777;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIG
# ============================================================

OKX_BASE = "https://www.okx.com"


# ============================================================
# STATIC FUTURES UNIVERSE
# ============================================================

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
    "QTUM"
]


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

TIMEFRAME_CONFIG = {

    "15m": {
        "bar": "15m",
        "minutes": 15
    },

    "1H": {
        "bar": "1H",
        "minutes": 60
    }

}


# ============================================================
# HELPER
# ============================================================

def timeframe_bars_for_hours(hours, timeframe):

    minutes = TIMEFRAME_CONFIG[timeframe]["minutes"]

    return max(
        1,
        int(round(hours * 60 / minutes))
    )


def format_horizon(horizon_bars, timeframe):

    hours = (
        horizon_bars
        * TIMEFRAME_CONFIG[timeframe]["minutes"]
        / 60
    )

    return (
        f"{horizon_bars} candle / "
        f"{hours:g} jam"
    )


def safe_float(value):

    try:
        return float(value)

    except Exception:

        return np.nan


# ============================================================
# NORMAL CDF
# Tidak membutuhkan scipy
# ============================================================

def normal_cdf(x):

    return (
        0.5
        * (
            1
            + math.erf(
                x / math.sqrt(2)
            )
        )
    )


# ============================================================
# CONFIDENCE ENGINE
# ============================================================

def calculate_confidence(
    prediction,
    uncertainty
):

    uncertainty = max(
        abs(float(uncertainty)),
        1e-9
    )

    prediction = float(prediction)

    # --------------------------------------------------------
    # Signal / Noise Ratio
    # --------------------------------------------------------

    ratio = (
        abs(prediction)
        / uncertainty
    )

    # --------------------------------------------------------
    # Statistik probability
    #
    # Ini bukan probabilitas profit.
    # Ini hanya probabilitas bahwa forecast
    # berada di sisi yang benar terhadap 0
    # berdasarkan asumsi Gaussian.
    # --------------------------------------------------------

    probability = normal_cdf(ratio)

    probability = np.clip(
        probability,
        0.50,
        0.9999
    )

    # --------------------------------------------------------
    # Confidence classification
    # --------------------------------------------------------

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
        float(probability),
        label
    )


# ============================================================
# OKX CANDLE DATA
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False
)
def get_okx_candles(
    symbol,
    bar="15m",
    limit=260
):

    inst_id = (
        f"{symbol}-USDT-SWAP"
    )

    url = (
        f"{OKX_BASE}"
        f"/api/v5/market/candles"
    )

    params = {

        "instId": inst_id,

        "bar": bar,

        "limit": str(limit)

    }

    response = requests.get(

        url,

        params=params,

        timeout=15,

        headers={
            "User-Agent":
            "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "0":

        raise RuntimeError(

            f"OKX code="
            f"{data.get('code')} "
            f"msg="
            f"{data.get('msg')}"

        )

    rows = data.get(
        "data",
        []
    )

    if not rows:

        raise RuntimeError(
            "No candle data"
        )


    # ========================================================
    # OKX CANDLE FORMAT
    #
    # [
    # timestamp,
    # open,
    # high,
    # low,
    # close,
    # volume,
    # volCcy,
    # volCcyQuote,
    # confirm
    # ]
    # ========================================================

    df = pd.DataFrame(

        rows,

        columns=[

            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vol_ccy",
            "vol_quote",
            "confirm"

        ]
    )


    # ========================================================
    # NUMERIC
    # ========================================================

    for col in [

        "open",
        "high",
        "low",
        "close",
        "volume"

    ]:

        df[col] = pd.to_numeric(

            df[col],

            errors="coerce"

        )


    df["ts"] = pd.to_numeric(

        df["ts"],

        errors="coerce"

    )


    # ========================================================
    # DATETIME
    # ========================================================

    df["datetime"] = pd.to_datetime(

        df["ts"],

        unit="ms",

        utc=True

    )


    df = df.sort_values(
        "datetime"
    )


    # ========================================================
    # ONLY CONFIRMED CANDLE
    # ========================================================

    if "confirm" in df.columns:

        df = df[
            df["confirm"]
            .astype(str)
            == "1"
        ]


    # ========================================================
    # CLEAN
    # ========================================================

    df = df.dropna(

        subset=[

            "open",
            "high",
            "low",
            "close",
            "volume"

        ]

    ).reset_index(
        drop=True
    )


    if len(df) < 120:

        raise RuntimeError(

            f"Insufficient candles: "
            f"{len(df)}"

        )


    return df


# ============================================================
# RSI
# ============================================================

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
        / avg_loss.replace(
            0,
            np.nan
        )
    )


    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )


    return rsi


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(
    df,
    timeframe
):

    x = df.copy()

    close = x["close"]

    volume = x["volume"]


    # ========================================================
    # TIMEFRAME-AWARE MOMENTUM
    #
    # 15m:
    # 6H  = 24 candles
    # 12H = 48 candles
    # 24H = 96 candles
    #
    # 1H:
    # 6H  = 6 candles
    # 12H = 12 candles
    # 24H = 24 candles
    # ========================================================

    mom6_bars = (
        timeframe_bars_for_hours(
            6,
            timeframe
        )
    )

    mom12_bars = (
        timeframe_bars_for_hours(
            12,
            timeframe
        )
    )

    mom24_bars = (
        timeframe_bars_for_hours(
            24,
            timeframe
        )
    )


    # ========================================================
    # RSI
    # ========================================================

    x["rsi14"] = calculate_rsi(
        close,
        14
    )


    # ========================================================
    # MOMENTUM
    # ========================================================

    x["mom_6"] = (

        close
        / close.shift(
            mom6_bars
        )
        - 1

    ) * 100


    x["mom_12"] = (

        close
        / close.shift(
            mom12_bars
        )
        - 1

    ) * 100


    x["mom_24"] = (

        close
        / close.shift(
            mom24_bars
        )
        - 1

    ) * 100


    # ========================================================
    # EMA
    # ========================================================

    x["ema12"] = close.ewm(

        span=12,

        adjust=False

    ).mean()


    x["ema24"] = close.ewm(

        span=24,

        adjust=False

    ).mean()


    # ========================================================
    # EMA DISTANCE
    # ========================================================

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


    # ========================================================
    # VOLUME Z
    # ========================================================

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

        (
            volume
            - volume_mean
        )
        /
        volume_std.replace(
            0,
            np.nan
        )

    )


    # ========================================================
    # VOLATILITY
    # ========================================================

    x["atr_pct"] = (

        (
            x["high"]
            - x["low"]
        )
        /
        close

    ) * 100


    # ========================================================
    # EMA TREND
    # ========================================================

    x["ema12_24_spread"] = (

        x["ema12"]
        /
        x["ema24"]
        - 1

    ) * 100


    return x


# ============================================================
# GPR FEATURES
# ============================================================

FEATURES = [

    "rsi14",

    "mom_6",

    "mom_12",

    "mom_24",

    "volume_z",

    "ema12_dist",

    "ema24_dist",

    "ema12_24_spread",

    "atr_pct"

]


# ============================================================
# TRAIN GPR
# ============================================================

def train_gpr(
    df,
    timeframe,
    horizon_bars
):

    data = add_features(
        df,
        timeframe
    )


    # ========================================================
    # FUTURE RETURN
    # ========================================================

    data["future_return"] = (

        data["close"]
        .shift(
            -horizon_bars
        )
        /
        data["close"]
        - 1

    ) * 100


    # ========================================================
    # TRAINING DATA
    # ========================================================

    model_data = data.dropna(

        subset=FEATURES
        + ["future_return"]

    ).copy()


    if len(model_data) < 90:

        raise RuntimeError(

            f"Not enough training "
            f"data: {len(model_data)}"

        )


    # ========================================================
    # LIMIT TRAINING SIZE
    # ========================================================

    model_data = (
        model_data
        .tail(180)
    )


    X = model_data[
        FEATURES
    ].replace(

        [np.inf, -np.inf],

        np.nan

    )


    y = model_data[
        "future_return"
    ]


    valid = (

        X.notna().all(
            axis=1
        )
        &
        y.notna()

    )


    X = X.loc[valid]

    y = y.loc[valid]


    if len(X) < 70:

        raise RuntimeError(

            "Insufficient valid "
            "training rows"

        )


    # ========================================================
    # SCALE
    # ========================================================

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(X)
    )


    # ========================================================
    # GPR KERNEL
    # ========================================================

    kernel = (

        ConstantKernel(
            1.0,
            (1e-3, 1e3)
        )

        *

        RBF(
            length_scale=1.0,
            length_scale_bounds=(
                1e-2,
                1e2
            )
        )

        +

        WhiteKernel(
            noise_level=0.5,
            noise_level_bounds=(
                1e-5,
                10
            )
        )

    )


    model = GaussianProcessRegressor(

        kernel=kernel,

        normalize_y=True,

        n_restarts_optimizer=0,

        random_state=42

    )


    model.fit(

        X_scaled,

        y.values

    )


    # ========================================================
    # LATEST CANDLE
    # ========================================================

    latest = data.dropna(

        subset=FEATURES

    ).iloc[-1]


    X_latest = (

        latest[FEATURES]
        .values
        .reshape(1, -1)

    )


    X_latest_scaled = (

        scaler.transform(
            X_latest
        )

    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction, std = model.predict(

        X_latest_scaled,

        return_std=True

    )


    prediction = float(
        prediction[0]
    )

    uncertainty = float(
        std[0]
    )


    return (

        prediction,
        uncertainty,
        latest

    )


# ============================================================
# GPR SCORE
# ============================================================

def calculate_score(

    prediction,

    uncertainty,

    latest

):

    # ========================================================
    # CONFIDENCE
    # ========================================================

    (
        confidence_ratio,
        confidence_probability,
        confidence_label

    ) = calculate_confidence(

        prediction,

        uncertainty

    )


    # ========================================================
    # MAGNITUDE
    #
    # Forecast 5% atau lebih = full component
    # ========================================================

    magnitude = np.clip(

        abs(prediction)
        / 5.0,

        0,
        1

    )


    # ========================================================
    # CONFIDENCE COMPONENT
    #
    # 0.5 ratio = 0
    # 3.0 ratio = 100%
    # ========================================================

    confidence_component = np.clip(

        (
            confidence_ratio
            - 0.5
        )
        /
        2.5,

        0,
        1

    )


    # ========================================================
    # RSI
    # ========================================================

    rsi = safe_float(
        latest["rsi14"]
    )


    if prediction > 0:

        if 45 <= rsi <= 70:

            rsi_component = 1.0

        elif (
            35 <= rsi < 45
            or
            70 < rsi <= 78
        ):

            rsi_component = 0.75

        else:

            rsi_component = 0.40

    else:

        if 30 <= rsi <= 55:

            rsi_component = 1.0

        elif (
            22 <= rsi < 30
            or
            55 < rsi <= 65
        ):

            rsi_component = 0.75

        else:

            rsi_component = 0.40


    # ========================================================
    # MOMENTUM ALIGNMENT
    # ========================================================

    momentum_values = [

        safe_float(
            latest["mom_6"]
        ),

        safe_float(
            latest["mom_12"]
        ),

        safe_float(
            latest["mom_24"]
        )

    ]


    avg_momentum = np.nanmean(
        momentum_values
    )


    if prediction > 0:

        if avg_momentum > 0.10:

            momentum_alignment = 1.0

        elif avg_momentum >= 0:

            momentum_alignment = 0.75

        else:

            momentum_alignment = 0.35

    else:

        if avg_momentum < -0.10:

            momentum_alignment = 1.0

        elif avg_momentum <= 0:

            momentum_alignment = 0.75

        else:

            momentum_alignment = 0.35


    # ========================================================
    # FINAL SCORE
    #
    # Magnitude      35
    # Confidence     30
    # RSI            15
    # Momentum       20
    # ========================================================

    score = (

        magnitude * 35

        +

        confidence_component * 30

        +

        rsi_component * 15

        +

        momentum_alignment * 20

    )


    score = float(
        np.clip(
            score,
            0,
            100
        )
    )


    return (

        score,

        confidence_ratio,

        confidence_probability,

        confidence_label

    )


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_symbol(

    symbol,

    timeframe,

    horizon_bars

):

    bar = TIMEFRAME_CONFIG[
        timeframe
    ]["bar"]


    df = get_okx_candles(

        symbol,

        bar=bar,

        limit=260

    )


    (
        prediction,
        uncertainty,
        latest

    ) = train_gpr(

        df,

        timeframe,

        horizon_bars

    )


    (
        score,
        confidence_ratio,
        confidence_probability,
        confidence_label

    ) = calculate_score(

        prediction,

        uncertainty,

        latest

    )


    # ========================================================
    # SIGNAL
    # ========================================================

    signal = (

        "LONG"

        if prediction >= 0

        else

        "SHORT"

    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "COIN":
            symbol,

        "SIGNAL":
            signal,

        "GPR SCORE":
            score,

        "GPR RETURN":
            prediction,

        "UNCERTAINTY":
            uncertainty,

        "CONFIDENCE":
            confidence_ratio,

        "CONF %":
            confidence_probability * 100,

        "CONF LABEL":
            confidence_label,

        "PRICE":
            float(
                latest["close"]
            ),

        "RSI":
            float(
                latest["rsi14"]
            ),

        "MOM 6H":
            float(
                latest["mom_6"]
            ),

        "MOM 12H":
            float(
                latest["mom_12"]
            ),

        "MOM 24H":
            float(
                latest["mom_24"]
            ),

        "VOL Z":
            float(
                latest["volume_z"]
            ),

        "EMA12 DIST":
            float(
                latest["ema12_dist"]
            ),

        "EMA24 DIST":
            float(
                latest["ema24_dist"]
            ),

        "CANDLE":
            latest["datetime"]

    }


    return result


# ============================================================
# RUN SCAN
# ============================================================

def run_scan(

    symbols,

    timeframe,

    horizon_bars

):

    results = []

    errors = []


    progress = st.progress(0)

    status = st.empty()


    total = len(symbols)


    # ========================================================
    # SEQUENTIAL
    #
    # Sengaja dibuat stabil untuk Streamlit Cloud.
    # ========================================================

    for i, symbol in enumerate(
        symbols
    ):

        status.write(

            f"Scanning "
            f"{symbol} "
            f"({i + 1}/{total})..."

        )


        try:

            result = analyze_symbol(

                symbol,

                timeframe,

                horizon_bars

            )

            results.append(
                result
            )


        except Exception as e:

            errors.append({

                "Symbol":
                    symbol,

                "Status":
                    str(e)

            })


        progress.progress(

            int(
                (
                    i + 1
                )
                /
                total
                * 100
            )

        )


        time.sleep(0.03)


    progress.empty()

    status.empty()


    return (
        results,
        errors
    )


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(

    title,

    value,

    subtitle=""

):

    with st.container(
        border=True
    ):

        st.markdown(

            f"""
            <div class="metric-title">
                {title}
            </div>
            """,

            unsafe_allow_html=True

        )


        st.markdown(

            f"""
            <div class="metric-value">
                {value}
            </div>
            """,

            unsafe_allow_html=True

        )


        if subtitle:

            st.caption(
                subtitle
            )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Scanner Settings"
)


# ============================================================
# TIMEFRAME
# ============================================================

timeframe = st.sidebar.selectbox(

    "Timeframe",

    [
        "15m",
        "1H"
    ],

    index=0

)


# ============================================================
# HORIZON
# ============================================================

horizon_options = [

    4,
    6,
    8,
    10,
    12,
    16,
    20

]


horizon = st.sidebar.selectbox(

    "Horizon GPR",

    horizon_options,

    index=3,

    format_func=lambda x:
        format_horizon(
            x,
            timeframe
        )

)


# ============================================================
# COIN COUNT
# ============================================================

coin_count = st.sidebar.selectbox(

    "Jumlah coin",

    [
        20,
        30,
        50,
        79
    ],

    index=3

)


# ============================================================
# MINIMUM SCORE
# ============================================================

minimum_score = st.sidebar.slider(

    "Minimum GPR Score",

    min_value=50,

    max_value=95,

    value=70,

    step=1

)


# ============================================================
# MINIMUM CONFIDENCE
# ============================================================

min_confidence = st.sidebar.slider(

    "Minimum Confidence",

    min_value=0.5,

    max_value=4.0,

    value=1.5,

    step=0.1

)


st.sidebar.caption(

    "Confidence = "
    "|Forecast| ÷ Uncertainty"

)


st.sidebar.divider()


# ============================================================
# BUTTON
# ============================================================

scan_button = st.sidebar.button(

    "🚀 SCAN FUTURES",

    type="primary",

    use_container_width=True

)


clear_button = st.sidebar.button(

    "🗑️ CLEAR CACHE",

    use_container_width=True

)


# ============================================================
# CLEAR
# ============================================================

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
    f"{timeframe} · "
    f"Horizon "
    f"{format_horizon(horizon, timeframe)} · "
    f"GPR sebagai mesin ranking"

)


# ============================================================
# INFO
# ============================================================

st.info(

    "Public market data OKX tanpa API key. "
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


    with st.spinner(

        f"Menjalankan GPR "
        f"Futures Scanner "
        f"{timeframe}..."

    ):

        (
            results,
            errors

        ) = run_scan(

            selected_symbols,

            timeframe,

            horizon

        )


    elapsed = (
        time.time()
        - start_time
    )


    if results:

        df_result = pd.DataFrame(
            results
        )


        # ====================================================
        # SORT
        # ====================================================

        df_result = (

            df_result

            .sort_values(

                [
                    "GPR SCORE",
                    "CONFIDENCE"

                ],

                ascending=[
                    False,
                    False
                ]

            )

            .reset_index(
                drop=True
            )

        )


        df_result["Rank"] = (

            np.arange(
                len(df_result)
            )
            + 1

        )


        # ====================================================
        # SAVE SESSION
        # ====================================================

        st.session_state[
            "results"
        ] = df_result


        st.session_state[
            "errors"
        ] = pd.DataFrame(
            errors
        )


        st.session_state[
            "timeframe"
        ] = timeframe


        st.session_state[
            "horizon"
        ] = horizon


        st.session_state[
            "elapsed"
        ] = elapsed


    else:

        st.error(

            "Tidak ada coin "
            "yang berhasil dianalisis."

        )


        if errors:

            st.dataframe(

                pd.DataFrame(
                    errors
                ),

                use_container_width=True,

                hide_index=True

            )


# ============================================================
# SHOW RESULTS
# ============================================================

if "results" in st.session_state:

    df = (
        st.session_state[
            "results"
        ].copy()
    )


    errors_df = (

        st.session_state.get(

            "errors",

            pd.DataFrame()

        )

    )


    active_tf = (

        st.session_state[
            "timeframe"
        ]

    )


    active_horizon = (

        st.session_state[
            "horizon"
        ]

    )


    elapsed = (

        st.session_state[
            "elapsed"
        ]

    )


    # ========================================================
    # CANDIDATES
    #
    # Dua syarat:
    # 1. Score memenuhi threshold
    # 2. Confidence memenuhi threshold
    # ========================================================

    candidates = df[

        (
            df["GPR SCORE"]
            >= minimum_score
        )

        &

        (
            df["CONFIDENCE"]
            >= min_confidence
        )

    ].copy()


    # ========================================================
    # SUMMARY
    # ========================================================

    st.success(

        f"Scan selesai: "
        f"{len(df)} berhasil, "
        f"{len(errors_df)} gagal "
        f"dalam {elapsed:.1f} detik."

    )


    # ========================================================
    # TOP INFO
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        metric_card(

            "UNIVERSE",

            len(df),

            "USDT Perpetual"

        )


    with c2:

        metric_card(

            "TIMEFRAME",

            active_tf,

            "Primary timeframe"

        )


    with c3:

        horizon_hours = (

            active_horizon
            *
            TIMEFRAME_CONFIG[
                active_tf
            ]["minutes"]
            /
            60

        )


        metric_card(

            "HORIZON",

            f"{active_horizon} candle",

            f"≈ {horizon_hours:g} jam"

        )


    with c4:

        metric_card(

            "PROVIDER",

            "OKX",

            "Public market data"

        )


    st.divider()


    # ========================================================
    # SECOND SUMMARY
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        metric_card(

            "BERHASIL DIANALISIS",

            len(df)

        )


    with c2:

        score_pass = len(

            df[
                df["GPR SCORE"]
                >= minimum_score
            ]

        )


        metric_card(

            "LOLOS SCORE",

            score_pass,

            f"≥ {minimum_score}"

        )


    with c3:

        metric_card(

            "KANDIDAT",

            len(candidates),

            (
                f"Score ≥ "
                f"{minimum_score} + "
                f"Confidence ≥ "
                f"{min_confidence:.1f}"
            )

        )


    with c4:

        if not candidates.empty:

            top = candidates.iloc[0]


        else:

            top = df.iloc[0]


        metric_card(

            "TOP GPR",

            top["COIN"],

            (
                f"{top['SIGNAL']} · "
                f"Score "
                f"{top['GPR SCORE']:.1f}"
            )

        )


    # ========================================================
    # CONFIDENCE GUIDE
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Cara Membaca Confidence"
    )


    st.markdown(

        """
### Confidence = |Forecast| ÷ Uncertainty

Angka ini adalah **signal-to-noise ratio**.

Artinya:

- **Forecast** = prediksi return GPR
- **Uncertainty** = tingkat ketidakpastian prediksi
- **Confidence** = seberapa besar forecast dibanding noise model

| Confidence | Interpretasi |
|---:|---|
| **< 0.50** | 🔴 SANGAT LEMAH |
| **0.50 – 0.99** | 🟠 LEMAH |
| **1.00 – 1.49** | 🟡 SEDANG |
| **1.50 – 1.99** | 🟢 BAIK |
| **2.00 – 2.49** | 🟢 KUAT |
| **2.50 – 2.99** | 🟢 SANGAT KUAT |
| **≥ 3.00** | 🔵 EKSTREM |

**Contoh:**

Forecast = **+4.53%**  
Uncertainty = **1.94%**

Maka:

**4.53 ÷ 1.94 = 2.33**

Jadi Confidence = **2.33 → KUAT**.

Tetapi ini **bukan berarti peluang profit 2.33 kali** dan bukan jaminan trade akan profit.
        """

    )


    # ========================================================
    # RANKING
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 GPR Futures Ranking"
    )


    display_df = df[

        [

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
            "VOL Z"

        ]

    ].copy()


    # ========================================================
    # FORMAT
    # ========================================================

    display_df[
        "GPR SCORE"
    ] = display_df[
        "GPR SCORE"
    ].round(1)


    display_df[
        "GPR RETURN"
    ] = display_df[
        "GPR RETURN"
    ].map(

        lambda x:
        f"{x:+.2f}%"

    )


    display_df[
        "UNCERTAINTY"
    ] = display_df[
        "UNCERTAINTY"
    ].map(

        lambda x:
        f"{x:.2f}%"

    )


    display_df[
        "CONFIDENCE"
    ] = display_df[
        "CONFIDENCE"
    ].map(

        lambda x:
        f"{x:.2f}"

    )


    display_df[
        "CONF %"
    ] = display_df[
        "CONF %"
    ].map(

        lambda x:
        f"{x:.1f}%"

    )


    display_df[
        "PRICE"
    ] = display_df[
        "PRICE"
    ].map(

        lambda x:
        f"{x:.8g}"

    )


    display_df[
        "RSI"
    ] = display_df[
        "RSI"
    ].round(1)


    for col in [

        "MOM 6H",
        "MOM 12H",
        "MOM 24H"

    ]:

        display_df[col] = (
            display_df[col]
            .map(
                lambda x:
                f"{x:+.2f}%"
            )
        )


    display_df[
        "VOL Z"
    ] = display_df[
        "VOL Z"
    ].round(2)


    # ========================================================
    # DATAFRAME
    # ========================================================

    st.dataframe(

        display_df,

        use_container_width=True,

        hide_index=True,

        column_config={

            "Rank":
                st.column_config.NumberColumn(
                    "🏆 Rank"
                ),

            "COIN":
                st.column_config.TextColumn(
                    "COIN"
                ),

            "SIGNAL":
                st.column_config.TextColumn(
                    "SIGNAL"
                ),

            "GPR SCORE":
                st.column_config.NumberColumn(
                    "GPR SCORE"
                ),

            "GPR RETURN":
                st.column_config.TextColumn(
                    "GPR RETURN"
                ),

            "UNCERTAINTY":
                st.column_config.TextColumn(
                    "UNCERTAINTY"
                ),

            "CONFIDENCE":
                st.column_config.TextColumn(
                    "CONFIDENCE"
                ),

            "CONF %":
                st.column_config.TextColumn(
                    "CONF %"
                ),

            "CONF LABEL":
                st.column_config.TextColumn(
                    "CONFIDENCE LEVEL"
                )

        }

    )


    # ========================================================
    # TOP SIGNALS
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Top GPR Signals"
    )


    if not candidates.empty:

        top_df = (
            candidates
            .head(6)
        )

    else:

        top_df = (
            df.head(6)
        )


    columns = st.columns(2)


    for idx, (_, row) in enumerate(
        top_df.iterrows()
    ):

        with columns[
            idx % 2
        ]:

            with st.container(
                border=True
            ):

                # ============================================
                # SIGNAL
                # ============================================

                if row["SIGNAL"] == "LONG":

                    signal_text = (
                        "🟢 LONG"
                    )

                else:

                    signal_text = (
                        "🔴 SHORT"
                    )


                st.markdown(
                    f"### "
                    f"{row['COIN']} · "
                    f"{signal_text}"
                )


                a, b, c = st.columns(3)


                with a:

                    st.metric(

                        "GPR Score",

                        f"{
                            row['GPR SCORE']
                            :.1f
                        }"

                    )


                with b:

                    st.metric(

                        "Forecast",

                        f"{
                            row['GPR RETURN']
                            :+.2f
                        }%"

                    )


                with c:

                    st.metric(

                        "Confidence",

                        f"{
                            row['CONFIDENCE']
                            :.2f
                        }"

                    )


                st.caption(

                    f"Level: "
                    f"{row['CONF LABEL']} · "

                    f"CONF statistik: "
                    f"{row['CONF %']:.1f}% · "

                    f"Uncertainty: "
                    f"{row['UNCERTAINTY']:.2f}%"

                )


    # ========================================================
    # DETAIL ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 Detail Analysis"
    )


    selected_coin = st.selectbox(

        "Pilih coin",

        df["COIN"].tolist()

    )


    selected = df[

        df["COIN"]
        == selected_coin

    ].iloc[0]


    # ========================================================
    # PRIMARY DETAIL
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(

            "SIGNAL",

            selected["SIGNAL"]

        )


    with c2:

        st.metric(

            "GPR SCORE",

            f"{
                selected['GPR SCORE']
                :.1f
            }"

        )


    with c3:

        st.metric(

            "FORECAST",

            f"{
                selected['GPR RETURN']
                :+.2f
            }%"

        )


    with c4:

        st.metric(

            "UNCERTAINTY",

            f"{
                selected['UNCERTAINTY']
                :.2f
            }%"

        )


    # ========================================================
    # CONFIDENCE DETAIL
    # ========================================================

    st.divider()

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(

            "CONFIDENCE",

            f"{
                selected['CONFIDENCE']
                :.2f
            }",

            selected["CONF LABEL"]

        )


    with c2:

        st.metric(

            "CONF %",

            f"{
                selected['CONF %']
                :.1f
            }%",

            "Statistical estimate"

        )


    with c3:

        st.metric(

            "RSI 14",

            f"{
                selected['RSI']
                :.1f
            }"

        )


    with c4:

        st.metric(

            "Volume Z",

            f"{
                selected['VOL Z']
                :.2f
            }"

        )


    # ========================================================
    # MOMENTUM
    # ========================================================

    st.divider()

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(

            "Harga",

            f"{
                selected['PRICE']
                :.8g
            }"

        )


    with c2:

        st.metric(

            "Momentum 6H",

            f"{
                selected['MOM 6H']
                :+.2f
            }%"

        )


    with c3:

        st.metric(

            "Momentum 12H",

            f"{
                selected['MOM 12H']
                :+.2f
            }%"

        )


    with c4:

        st.metric(

            "Momentum 24H",

            f"{
                selected['MOM 24H']
                :+.2f
            }%"

        )


    # ========================================================
    # EMA
    # ========================================================

    c1, c2 = st.columns(2)


    with c1:

        st.metric(

            "EMA12 Distance",

            f"{
                selected['EMA12 DIST']
                :+.2f
            }%"

        )


    with c2:

        st.metric(

            "EMA24 Distance",

            f"{
                selected['EMA24 DIST']
                :+.2f
            }%"

        )


    st.caption(

        "Last confirmed candle: "
        f"{selected['CANDLE']}"

    )


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    st.divider()


    with st.expander(
        "📋 ERROR / DIAGNOSTIC"
    ):

        if errors_df.empty:

            st.success(

                "Tidak ada error "
                "pada coin yang dianalisis."

            )

        else:

            st.dataframe(

                errors_df,

                use_container_width=True,

                hide_index=True

            )


    # ========================================================
    # EXPORT CSV
    # ========================================================

    st.divider()


    csv = df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(

        "⬇️ Download Ranking CSV",

        data=csv,

        file_name=(
            f"gpr_crypto_"
            f"{active_tf}.csv"
        ),

        mime="text/csv"

    )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.info(

        f"Belum ada hasil. "
        f"Tekan **SCAN FUTURES** "
        f"untuk menjalankan scanner "
        f"{timeframe}."

    )


    st.markdown(

        """
### GPR Crypto Futures Scanner

**Konfigurasi saat ini:**

- Primary timeframe: **15m**
- Secondary timeframe: **1H**
- Provider: **OKX Public Market Data**
- API key: **tidak diperlukan**
- Model: **Gaussian Process Regression**
- Signal: **LONG / SHORT**
- Ranking: **GPR Score**
- Forecast: **GPR Return**
- Risk: **Uncertainty**
- Confidence: **Forecast ÷ Uncertainty**
- Confidence %: **indikator statistik, bukan jaminan profit**

### Pembacaan Confidence

**1.00** → forecast hanya sekitar 1× uncertainty  
**1.50** → mulai menarik  
**2.00** → kuat  
**2.50** → sangat kuat  
**3.00+** → sangat tinggi, tetapi tetap harus divalidasi

"""

    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(

    "GPR Crypto Futures Scanner · "
    "USDT Perpetual · "
    "Public Market Data · "
    "Research Prototype"

)
