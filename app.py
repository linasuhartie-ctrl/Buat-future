import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="GPR Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2.2rem !important;
    }

    .small-muted {
        color: #6b7280;
        font-size: .85rem;
    }

    .signal-long {
        color: #0a9f5a;
        font-weight: 800;
    }

    .signal-short {
        color: #e53935;
        font-weight: 800;
    }

    .guide {
        padding: 16px;
        border: 1px solid #d9dee7;
        border-radius: 12px;
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANT
# ============================================================

OKX_BASE = "https://www.okx.com"


# ============================================================
# STATIC UNIVERSE
# ============================================================

# Static list sengaja dipakai supaya scanner tidak bergantung
# pada endpoint instruments/exchangeInfo yang sebelumnya
# menghasilkan HTTP 403/451.

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


TIMEFRAMES = {
    "15m": {
        "bar": "15m",
        "minutes": 15,
    },
    "1H": {
        "bar": "1H",
        "minutes": 60,
    },
}


FEATURES = [
    "rsi14",
    "mom_6",
    "mom_12",
    "mom_24",
    "volume_z",
    "ema12_dist",
    "ema24_dist",
    "ema12_24_spread",
    "atr_pct",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def horizon_label(bars, timeframe):
    mins = bars * TIMEFRAMES[timeframe]["minutes"]

    if mins >= 60:
        return f"{bars} candle / {mins / 60:g} jam"

    return f"{bars} candle / {mins:g} menit"


def bars_for_hours(hours, timeframe):
    return max(
        1,
        int(
            round(
                hours
                * 60
                / TIMEFRAMES[timeframe]["minutes"]
            )
        ),
    )


def normal_cdf(x):
    return 0.5 * (
        1.0
        + math.erf(
            x / math.sqrt(2.0)
        )
    )


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_info(
    forecast,
    uncertainty,
):
    """
    Confidence = |forecast| / uncertainty.

    Ini adalah ukuran signal-to-noise,
    BUKAN probabilitas profit.
    """

    u = max(
        abs(float(uncertainty)),
        1e-9,
    )

    f = float(forecast)

    ratio = abs(f) / u

    probability = float(
        np.clip(
            normal_cdf(ratio),
            0.50,
            0.9999,
        )
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
        ratio,
        probability,
        label,
    )


def confidence_icon(label):

    if label in (
        "SANGAT LEMAH",
        "LEMAH",
    ):
        return "🔴"

    if label == "SEDANG":
        return "🟡"

    if label in (
        "BAIK",
        "KUAT",
        "SANGAT KUAT",
    ):
        return "🟢"

    return "🔵"


# ============================================================
# OKX PUBLIC MARKET DATA
# ============================================================

@st.cache_data(
    ttl=120,
    show_spinner=False,
)
def get_okx_candles(
    symbol,
    bar,
    limit=220,
):

    inst_id = f"{symbol}-USDT-SWAP"

    url = (
        f"{OKX_BASE}"
        "/api/v5/market/candles"
    )

    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": str(limit),
    }

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "GPR-Crypto-Scanner"
    }

    last_error = "Unknown error"

    # Retry hanya 2x.
    # Tujuannya supaya coin bermasalah
    # tidak menahan seluruh scanner terlalu lama.

    for attempt in range(2):

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=(
                    3.5,
                    10,
                ),
            )

            response.raise_for_status()

            payload = response.json()

            if str(
                payload.get("code")
            ) != "0":

                raise RuntimeError(
                    "OKX code="
                    f"{payload.get('code')} "
                    f"msg={payload.get('msg')}"
                )

            rows = (
                payload.get("data")
                or []
            )

            if not rows:
                raise RuntimeError(
                    "OKX tidak "
                    "mengembalikan candle."
                )

            cols = [
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

            data = pd.DataFrame(
                rows,
                columns=cols,
            )

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:

                data[col] = pd.to_numeric(
                    data[col],
                    errors="coerce",
                )

            data["ts"] = pd.to_numeric(
                data["ts"],
                errors="coerce",
            )

            data["datetime"] = pd.to_datetime(
                data["ts"],
                unit="ms",
                utc=True,
                errors="coerce",
            )

            # Hanya candle yang sudah confirmed.
            if "confirm" in data.columns:

                data = data[
                    data["confirm"]
                    .astype(str)
                    == "1"
                ]

            data = (
                data
                .sort_values(
                    "datetime"
                )
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
                .reset_index(
                    drop=True
                )
            )

            if len(data) < 120:

                raise RuntimeError(
                    "Data candle kurang: "
                    f"{len(data)} bar valid."
                )

            return data

        except Exception as exc:

            last_error = str(exc)

            if attempt < 1:
                time.sleep(0.35)

    raise RuntimeError(
        last_error
    )


# ============================================================
# RSI
# ============================================================

def rsi(
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

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False,
        )
        .mean()
    )

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
# TECHNICAL FEATURES
# ============================================================

def add_features(
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
    x["rsi14"] = rsi(
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
    x["ema12"] = (
        close.ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    x["ema24"] = (
        close.ewm(
            span=24,
            adjust=False,
        )
        .mean()
    )

    # EMA distance
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

    x["ema12_24_spread"] = (
        x["ema12"]
        / x["ema24"]
        - 1
    ) * 100

    # Volume Z-score
    vmean = (
        volume
        .rolling(30)
        .mean()
    )

    vstd = (
        volume
        .rolling(30)
        .std()
    )

    x["volume_z"] = (
        volume - vmean
    ) / vstd.replace(
        0,
        np.nan,
    )

    # ========================================================
    # ATR
    # ========================================================

    prev_close = close.shift(1)

    tr1 = (
        x["high"]
        - x["low"]
    )

    tr2 = (
        x["high"]
        - prev_close
    ).abs()

    tr3 = (
        x["low"]
        - prev_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
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
# GPR TRAINING
# ============================================================

def train_gpr(
    df,
    timeframe,
    horizon_bars,
):

    data = add_features(
        df,
        timeframe,
    )

    # Target:
    # return beberapa candle ke depan.

    data["future_return"] = (
        data["close"]
        .shift(-horizon_bars)
        / data["close"]
        - 1
    ) * 100

    # ========================================================
    # OPTIMASI:
    # sebelumnya 180 training rows.
    # sekarang 150 rows.
    # ========================================================

    model_data = (
        data
        .dropna(
            subset=FEATURES
            + ["future_return"]
        )
        .tail(150)
        .copy()
    )

    if len(model_data) < 60:

        raise RuntimeError(
            "Training data tidak "
            "cukup: "
            f"{len(model_data)} bar."
        )

    X = model_data[
        FEATURES
    ].replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    y = model_data[
        "future_return"
    ]

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid]

    y = y.loc[valid]

    if len(X) < 60:

        raise RuntimeError(
            "Training rows valid "
            f"< 60. Saat ini {len(X)}."
        )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    scaler = StandardScaler()

    Xs = scaler.fit_transform(X)

    # ========================================================
    # GPR KERNEL
    # ========================================================

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

    # ========================================================
    # OPTIMASI UTAMA
    # ========================================================

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,

        # Tidak melakukan restart optimizer.
        # Ini sangat mengurangi CPU time.
        n_restarts_optimizer=0,

        random_state=42,
    )

    model.fit(
        Xs,
        y.to_numpy(),
    )

    # ========================================================
    # LATEST FEATURE
    # ========================================================

    latest = (
        data
        .dropna(
            subset=FEATURES
        )
        .iloc[-1]
    )

    xl = (
        latest[
            FEATURES
        ]
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
            scaler.transform(xl),
            return_std=True,
        )
    )

    return (
        float(prediction[0]),
        max(
            float(std[0]),
            1e-6,
        ),
        latest,
    )


# ============================================================
# GPR SCORE
# ============================================================

def calculate_score(
    prediction,
    uncertainty,
    latest,
):

    (
        ratio,
        probability,
        label,
    ) = confidence_info(
        prediction,
        uncertainty,
    )

    # Forecast 5% dianggap magnitude
    # maksimum untuk score.

    magnitude = float(
        np.clip(
            abs(prediction)
            / 5.0,
            0,
            1,
        )
    )

    # Confidence:
    # 0.5 -> 0
    # 3.0 -> 100

    confidence_component = float(
        np.clip(
            (ratio - 0.5)
            / 2.5,
            0,
            1,
        )
    )

    # ========================================================
    # RSI COMPONENT
    # ========================================================

    r = float(
        latest["rsi14"]
    )

    if prediction >= 0:

        if 45 <= r <= 70:
            rsi_component = 1.0

        elif (
            35 <= r < 45
            or 70 < r <= 78
        ):
            rsi_component = 0.75

        else:
            rsi_component = 0.40

    else:

        if 30 <= r <= 55:
            rsi_component = 1.0

        elif (
            22 <= r < 30
            or 55 < r <= 65
        ):
            rsi_component = 0.75

        else:
            rsi_component = 0.40

    # ========================================================
    # MOMENTUM COMPONENT
    # ========================================================

    momentum = float(
        np.nanmean(
            [
                latest["mom_6"],
                latest["mom_12"],
                latest["mom_24"],
            ]
        )
    )

    if prediction >= 0:

        if momentum > 0.10:
            momentum_component = 1.0

        elif momentum >= 0:
            momentum_component = 0.75

        else:
            momentum_component = 0.35

    else:

        if momentum < -0.10:
            momentum_component = 1.0

        elif momentum <= 0:
            momentum_component = 0.75

        else:
            momentum_component = 0.35

    # ========================================================
    # FINAL SCORE
    # ========================================================

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
        ratio,
        probability,
        label,
    )


# ============================================================
# SINGLE COIN ANALYSIS
# ============================================================

def _analyze_symbol(
    symbol,
    timeframe,
    horizon_bars,
):

    bar = TIMEFRAMES[
        timeframe
    ]["bar"]

    candles = get_okx_candles(
        symbol,
        bar,
        220,
    )

    (
        forecast,
        uncertainty,
        latest,
    ) = train_gpr(
        candles,
        timeframe,
        horizon_bars,
    )

    (
        score,
        confidence,
        conf_probability,
        conf_label,
    ) = calculate_score(
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

        "CONF %":
            conf_probability * 100,

        "CONF LABEL": conf_label,

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
            latest["datetime"],
    }


# ============================================================
# CACHE SINGLE COIN
# ============================================================

@st.cache_data(
    ttl=120,
    show_spinner=False,
)
def analyze_symbol(
    symbol,
    timeframe,
    horizon_bars,
):

    return _analyze_symbol(
        symbol,
        timeframe,
        horizon_bars,
    )


# ============================================================
# RUN SCAN
# ============================================================

def run_scan(
    symbols,
    timeframe,
    horizon_bars,
    workers,
):

    results = []

    errors = []

    progress = st.progress(0)

    status = st.empty()

    total = len(symbols)

    # ========================================================
    # THREAD POOL
    # ========================================================
    #
    # Thread digunakan terutama untuk I/O HTTP.
    #
    # Worker sengaja dibatasi melalui sidebar.
    #

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
                "Scanning "
                f"{symbol} "
                f"({completed}/{total})..."
            )

            try:

                results.append(
                    future.result()
                )

            except Exception as exc:

                errors.append(
                    {
                        "Symbol":
                            symbol,
                        "Status":
                            str(exc),
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
# UI HELPERS
# ============================================================

def metric_card(
    title,
    value,
    subtitle="",
):

    with st.container(
        border=True
    ):

        st.caption(title)

        st.markdown(
            f"## {value}"
        )

        if subtitle:

            st.caption(
                subtitle
            )


def format_result_table(
    df
):

    out = df[
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
            "VOL Z",
        ]
    ].copy()

    out["GPR SCORE"] = (
        out["GPR SCORE"]
        .round(1)
    )

    out["GPR RETURN"] = (
        out["GPR RETURN"]
        .map(
            lambda x:
                f"{x:+.2f}%"
        )
    )

    out["UNCERTAINTY"] = (
        out["UNCERTAINTY"]
        .map(
            lambda x:
                f"{x:.2f}%"
        )
    )

    out["CONFIDENCE"] = (
        out["CONFIDENCE"]
        .map(
            lambda x:
                f"{x:.2f}"
        )
    )

    out["CONF %"] = (
        out["CONF %"]
        .map(
            lambda x:
                f"{x:.1f}%"
        )
    )

    out["PRICE"] = (
        out["PRICE"]
        .map(
            lambda x:
                f"{x:.8g}"
        )
    )

    out["RSI"] = (
        out["RSI"]
        .round(1)
    )

    for col in [
        "MOM 6H",
        "MOM 12H",
        "MOM 24H",
    ]:

        out[col] = (
            out[col]
            .map(
                lambda x:
                    f"{x:+.2f}%"
            )
        )

    out["VOL Z"] = (
        out["VOL Z"]
        .round(2)
    )

    return out


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
        "1H",
    ],
    index=0,
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
    20,
]

horizon = st.sidebar.selectbox(
    "Horizon GPR",
    horizon_options,
    index=3,
    format_func=lambda x:
        horizon_label(
            x,
            timeframe,
        ),
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
        79,
    ],
    index=3,
)


# ============================================================
# MINIMUM SCORE
# ============================================================

minimum_score = st.sidebar.slider(
    "Minimum GPR Score",
    min_value=50,
    max_value=95,
    value=70,
    step=1,
)


# ============================================================
# MINIMUM CONFIDENCE
# ============================================================

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.5,
    max_value=4.0,
    value=1.5,
    step=0.1,
)


# ============================================================
# WORKERS
# ============================================================

workers = st.sidebar.slider(
    "Parallel workers",
    min_value=1,
    max_value=8,
    value=3,
    step=1,
)


st.sidebar.caption(
    "Confidence = "
    "|Forecast| ÷ Uncertainty"
)

st.sidebar.caption(
    "Default 3 worker untuk menjaga "
    "CPU Streamlit stabil saat GPR."
)


st.sidebar.divider()


# ============================================================
# BUTTON
# ============================================================

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
# CLEAR CACHE
# ============================================================

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

    st.session_state.pop(
        "scan_meta",
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
    "GPR sebagai mesin ranking"
)


st.info(
    f"Public market data OKX tanpa API key · "
    f"{timeframe} · "
    f"Horizon {horizon_label(horizon, timeframe)}. "
    "Universe memakai daftar static agar scanner "
    "tidak bergantung pada endpoint "
    "instruments/exchangeInfo."
)


# ============================================================
# SCAN
# ============================================================

if scan_button:

    selected_symbols = (
        UNIVERSE[:coin_count]
    )

    start = time.time()

    with st.spinner(
        "Menjalankan scanner "
        f"{timeframe} untuk "
        f"{len(selected_symbols)} coin..."
    ):

        (
            results,
            errors,
        ) = run_scan(
            selected_symbols,
            timeframe,
            horizon,
            workers,
        )

    elapsed = (
        time.time()
        - start
    )

    if results:

        df_result = pd.DataFrame(
            results
        )

        # ====================================================
        # RANKING
        # ====================================================
        #
        # Ranking utama:
        # 1. GPR SCORE
        # 2. CONFIDENCE
        #
        # Tidak dipaksa coin tertentu
        # menjadi nomor 1.
        #

        df_result = (
            df_result
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
            .reset_index(
                drop=True
            )
        )

        df_result.insert(
            0,
            "Rank",
            np.arange(
                1,
                len(df_result) + 1,
            ),
        )

        st.session_state[
            "results"
        ] = df_result

        st.session_state[
            "errors"
        ] = pd.DataFrame(
            errors
        )

        st.session_state[
            "scan_meta"
        ] = {
            "timeframe":
                timeframe,

            "horizon":
                horizon,

            "elapsed":
                elapsed,
        }

        st.rerun()

    else:

        st.error(
            "Tidak ada coin yang "
            "berhasil dianalisis."
        )

        if errors:

            st.dataframe(
                pd.DataFrame(
                    errors
                ),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# EMPTY STATE
# ============================================================

if "results" not in st.session_state:

    st.info(
        "Belum ada hasil. Tekan "
        "SCAN FUTURES untuk "
        f"menjalankan scanner "
        f"{timeframe}."
    )

    st.subheader(
        "📌 Konfigurasi"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "TIMEFRAME",
            timeframe,
            "Primary timeframe",
        )

    with c2:

        metric_card(
            "HORIZON",
            f"{horizon} candle",
            horizon_label(
                horizon,
                timeframe,
            ),
        )

    with c3:

        metric_card(
            "UNIVERSE",
            coin_count,
            "USDT Perpetual",
        )

    with c4:

        metric_card(
            "PROVIDER",
            "OKX",
            "Public market data",
        )

    st.subheader(
        "📊 Cara membaca Confidence"
    )

    st.markdown(
        """
        <div class="guide">

        <b>Confidence = |Forecast| ÷ Uncertainty</b>

        <br><br>

        Ini adalah rasio signal-to-noise model.
        Semakin besar nilainya, semakin besar forecast
        dibanding ketidakpastian model.

        <br><br>

        <b>&lt; 0.50</b> 🔴 Sangat Lemah
        &nbsp; |

        <b>0.50–0.99</b> 🟠 Lemah
        &nbsp; |

        <b>1.00–1.49</b> 🟡 Sedang
        &nbsp; |

        <b>1.50–1.99</b> 🟢 Baik
        &nbsp; |

        <b>2.00–2.49</b> 🟢 Kuat
        &nbsp; |

        <b>2.50–2.99</b> 🟢 Sangat Kuat
        &nbsp; |

        <b>≥ 3.00</b> 🔵 Ekstrem

        <br><br>

        <b>Catatan:</b>
        Confidence bukan probabilitas profit.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HASIL SCAN
# ============================================================

else:

    df = (
        st.session_state[
            "results"
        ].copy()
    )

    errors_df = (
        st.session_state.get(
            "errors",
            pd.DataFrame(),
        )
    )

    meta = (
        st.session_state[
            "scan_meta"
        ]
    )

    active_tf = (
        meta["timeframe"]
    )

    active_horizon = (
        meta["horizon"]
    )

    elapsed = (
        meta["elapsed"]
    )


    # ========================================================
    # FILTER CANDIDATES
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
    # SCAN STATUS
    # ========================================================

    st.success(
        f"Scan selesai: "
        f"{len(df)} berhasil, "
        f"{len(errors_df)} gagal "
        f"dalam {elapsed:.1f} detik."
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        metric_card(
            "UNIVERSE",
            len(df),
            "Coin berhasil dianalisis",
        )

    with c2:

        metric_card(
            "TIMEFRAME",
            active_tf,
            "Primary timeframe",
        )

    with c3:

        metric_card(
            "HORIZON",
            f"{active_horizon} candle",
            horizon_label(
                active_horizon,
                active_tf,
            ),
        )

    with c4:

        metric_card(
            "PROVIDER",
            "OKX",
            "Public market data",
        )


    st.divider()


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        metric_card(
            "BERHASIL DIANALISIS",
            len(df),
        )


    with c2:

        score_pass = int(
            (
                df["GPR SCORE"]
                >= minimum_score
            ).sum()
        )

        metric_card(
            "LOLOS SCORE",
            score_pass,
            f"≥ {minimum_score}",
        )


    with c3:

        metric_card(
            "KANDIDAT",
            len(candidates),
            f"Score ≥ {minimum_score} "
            f"+ Confidence ≥ "
            f"{min_confidence:.1f}",
        )


    with c4:

        if not candidates.empty:

            top = (
                candidates.iloc[0]
            )

        else:

            top = (
                df.iloc[0]
            )

        metric_card(
            "TOP GPR",
            top["COIN"],
            f"{top['SIGNAL']} · "
            f"Score "
            f"{top['GPR SCORE']:.1f}",
        )


    # ========================================================
    # RANKING
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 GPR Futures Ranking"
    )

    st.caption(
        "Ranking diurutkan berdasarkan "
        "GPR Score, lalu Confidence."
    )


    display_df = (
        format_result_table(
            df
        )
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=520,
        column_config={

            "Rank":
                st.column_config.NumberColumn(
                    "🏆 Rank",
                    width="small",
                ),

            "GPR SCORE":
                st.column_config.NumberColumn(
                    "GPR SCORE",
                ),

            "CONFIDENCE":
                st.column_config.TextColumn(
                    "CONFIDENCE",
                ),

            "CONF %":
                st.column_config.TextColumn(
                    "CONF %",
                ),

            "CONF LABEL":
                st.column_config.TextColumn(
                    "CONFIDENCE LEVEL",
                ),
        },
    )


    # ========================================================
    # TOP SIGNALS
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Top GPR Signals"
    )


    top_df = (
        candidates.head(6)
        if not candidates.empty
        else df.head(6)
    )


    if candidates.empty:

        st.warning(
            "Belum ada kandidat yang "
            "memenuhi filter. "
            "Turunkan Minimum GPR Score "
            "atau Minimum Confidence."
        )


    cards = st.columns(2)


    for i, (_, row) in enumerate(
        top_df.iterrows()
    ):

        with cards[
            i % 2
        ]:

            with st.container(
                border=True
            ):

                icon = (
                    "🟢 LONG"
                    if row["SIGNAL"]
                    == "LONG"
                    else "🔴 SHORT"
                )


                st.markdown(
                    f"### "
                    f"{row['COIN']} "
                    f"· {icon}"
                )


                a, b, c = (
                    st.columns(3)
                )


                with a:

                    st.metric(
                        "GPR Score",
                        f"{row['GPR SCORE']:.1f}",
                    )


                with b:

                    st.metric(
                        "Forecast",
                        f"{row['GPR RETURN']:+.2f}%",
                    )


                with c:

                    st.metric(
                        "Confidence",
                        f"{row['CONFIDENCE']:.2f}",
                    )


                st.caption(
                    f"{confidence_icon(row['CONF LABEL'])} "
                    f"{row['CONF LABEL']} · "
                    f"Confidence statistik "
                    f"{row['CONF %']:.1f}% · "
                    f"Uncertainty "
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
        df["COIN"].tolist(),
    )


    selected = (
        df[
            df["COIN"]
            == selected_coin
        ]
        .iloc[0]
    )


    # ========================================================
    # PRIMARY SIGNAL
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "SIGNAL",
            selected["SIGNAL"],
        )


    with c2:

        st.metric(
            "GPR SCORE",
            f"{selected['GPR SCORE']:.1f}",
        )


    with c3:

        st.metric(
            "FORECAST",
            f"{selected['GPR RETURN']:+.2f}%",
        )


    with c4:

        st.metric(
            "UNCERTAINTY",
            f"{selected['UNCERTAINTY']:.2f}%",
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "CONFIDENCE",
            f"{selected['CONFIDENCE']:.2f}",
            f"{confidence_icon(selected['CONF LABEL'])} "
            f"{selected['CONF LABEL']}",
        )


    with c2:

        st.metric(
            "CONF % STATISTIK",
            f"{selected['CONF %']:.1f}%",
            "Bukan win probability",
        )


    with c3:

        st.metric(
            "RSI 14",
            f"{selected['RSI']:.1f}",
        )


    with c4:

        st.metric(
            "VOLUME Z",
            f"{selected['VOL Z']:.2f}",
        )


    # ========================================================
    # TECHNICAL DATA
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "HARGA",
            f"{selected['PRICE']:.8g}",
        )


    with c2:

        st.metric(
            "MOMENTUM 6H",
            f"{selected['MOM 6H']:+.2f}%",
        )


    with c3:

        st.metric(
            "MOMENTUM 12H",
            f"{selected['MOM 12H']:+.2f}%",
        )


    with c4:

        st.metric(
            "MOMENTUM 24H",
            f"{selected['MOM 24H']:+.2f}%",
        )


    c1, c2 = (
        st.columns(2)
    )


    with c1:

        st.metric(
            "EMA12 DISTANCE",
            f"{selected['EMA12 DIST']:+.2f}%",
        )


    with c2:

        st.metric(
            "EMA24 DISTANCE",
            f"{selected['EMA24 DIST']:+.2f}%",
        )


    st.caption(
        "Last confirmed candle: "
        f"{selected['CANDLE']}"
    )


    # ========================================================
    # CONFIDENCE GUIDE
    # ========================================================

    st.divider()

    st.subheader(
        "📚 Confidence Guide"
    )


    st.markdown(
        """
        <div class="guide">

        <b>Rasio = |Forecast| / Uncertainty</b>

        <br><br>

        0.0–0.49 = 🔴 sangat lemah
        <br>

        0.50–0.99 = 🟠 lemah
        <br>

        1.00–1.49 = 🟡 sedang
        <br>

        1.50–1.99 = 🟢 baik
        <br>

        2.00–2.49 = 🟢 kuat
        <br>

        2.50–2.99 = 🟢 sangat kuat
        <br>

        ≥ 3.00 = 🔵 ekstrem

        <br><br>

        <b>Contoh:</b>
        Forecast +4.53% dan
        Uncertainty 1.94%
        → Confidence 2.33 → KUAT.

        <br><br>

        Angka ini menunjukkan
        kekuatan relatif model
        terhadap uncertainty,
        bukan jaminan harga akan
        bergerak sesuai forecast.

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # ERROR / DIAGNOSTIC
    # ========================================================

    st.divider()


    with st.expander(
        f"📋 ERROR / DIAGNOSTIC "
        f"({len(errors_df)} error)"
    ):

        if errors_df.empty:

            st.success(
                "Tidak ada error pada "
                "coin yang dianalisis."
            )

        else:

            st.dataframe(
                errors_df,
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # EXPORT
    # ========================================================

    st.divider()


    csv_data = (
        df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    st.download_button(
        "⬇️ Download Ranking CSV",
        data=csv_data,
        file_name=(
            f"gpr_crypto_{active_tf}.csv"
        ),
        mime="text/csv",
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
