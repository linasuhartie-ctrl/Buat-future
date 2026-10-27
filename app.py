import math
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
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
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-size: 2.35rem !important;
        margin-bottom: .15rem !important;
    }

    .long {
        color: #0a9f5a;
        font-weight: 800;
    }

    .short {
        color: #e53935;
        font-weight: 800;
    }

    .guide {
        padding: 14px 18px;
        border: 1px solid #ddd;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OKX
# ============================================================

OKX_BASE = "https://www.okx.com"


# ============================================================
# STATIC FUTURES UNIVERSE
# ============================================================

UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LINK", "DOT",
    "TRX", "SUI", "TON", "SHIB", "LTC", "BCH", "NEAR", "APT", "ICP", "FIL",
    "ATOM", "ETC", "ARB", "OP", "INJ", "IMX", "AAVE", "MKR", "RUNE", "UNI",
    "LDO", "STX", "SEI", "TIA", "JUP", "WIF", "PEPE", "BONK", "FLOKI", "WLD",
    "ENA", "ONDO", "PYTH", "JTO", "STRK", "NOT", "ORDI", "SATS", "SAND", "MANA",
    "AXS", "GALA", "EOS", "XLM", "VET", "ALGO", "HBAR", "FTM", "APE", "CRV",
    "COMP", "SNX", "DYDX", "1INCH", "KAS", "TAO", "FET", "RENDER", "MANTA", "DYM",
    "WOO", "JASMY", "KAVA", "MINA", "CELO", "ZRX", "QTUM",
]


# ============================================================
# TIMEFRAME
# ============================================================

TIMEFRAME_CONFIG = {
    "15m": {
        "bar": "15m",
        "minutes": 15,
    },
    "1H": {
        "bar": "1H",
        "minutes": 60,
    },
}


# ============================================================
# MODEL FEATURES
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
    "atr_pct",
]


# ============================================================
# FORWARD VALIDATION
# ============================================================

VALIDATION_POINTS = 8

FORWARD_HORIZONS = [5, 10]


# ============================================================
# HELPERS
# ============================================================

def timeframe_bars_for_hours(hours, timeframe):
    minutes = TIMEFRAME_CONFIG[timeframe]["minutes"]

    return max(
        1,
        int(round(hours * 60 / minutes)),
    )


def format_horizon(horizon_bars, timeframe):
    hours = (
        horizon_bars
        * TIMEFRAME_CONFIG[timeframe]["minutes"]
        / 60
    )

    return f"{horizon_bars} candle / {hours:g} jam"


def normal_cdf(x):
    return 0.5 * (
        1.0
        + math.erf(
            x / math.sqrt(2.0)
        )
    )


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(
    prediction,
    uncertainty,
):
    uncertainty = max(
        abs(float(uncertainty)),
        1e-9,
    )

    prediction = float(prediction)

    ratio = (
        abs(prediction)
        / uncertainty
    )

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
        float(ratio),
        probability,
        label,
    )


def confidence_color(label):

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
# OKX DATA
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False,
)
def get_okx_candles(
    symbol,
    bar,
    limit=260,
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
        "User-Agent": "Mozilla/5.0",
    }

    last_error = None

    for attempt in range(3):

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=15,
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get("code") != "0":

                raise RuntimeError(
                    "OKX code="
                    f"{payload.get('code')} "
                    "msg="
                    f"{payload.get('msg')}"
                )

            rows = payload.get(
                "data",
                [],
            )

            if not rows:
                raise RuntimeError(
                    "No candle data"
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

            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]:

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

            df = df.sort_values(
                "datetime"
            )

            if "confirm" in df.columns:

                df = df[
                    df["confirm"]
                    .astype(str)
                    == "1"
                ]

            df = df.dropna(
                subset=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ).reset_index(
                drop=True
            )

            if len(df) < 120:

                raise RuntimeError(
                    "Insufficient candles: "
                    f"{len(df)}"
                )

            return df

        except Exception as exc:

            last_error = exc

            if attempt < 2:
                time.sleep(
                    0.5
                    * (attempt + 1)
                )

    raise RuntimeError(
        str(last_error)
    )


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

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# FEATURES
# ============================================================

def add_features(
    df,
    timeframe,
):
    x = df.copy()

    close = x["close"]
    volume = x["volume"]

    mom6 = timeframe_bars_for_hours(
        6,
        timeframe,
    )

    mom12 = timeframe_bars_for_hours(
        12,
        timeframe,
    )

    mom24 = timeframe_bars_for_hours(
        24,
        timeframe,
    )

    x["rsi14"] = calculate_rsi(
        close,
        14,
    )

    x["mom_6"] = (
        close
        / close.shift(mom6)
        - 1
    ) * 100

    x["mom_12"] = (
        close
        / close.shift(mom12)
        - 1
    ) * 100

    x["mom_24"] = (
        close
        / close.shift(mom24)
        - 1
    ) * 100

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

    x["ema12_24_spread"] = (
        x["ema12"]
        / x["ema24"]
        - 1
    ) * 100

    volume_mean = (
        volume.rolling(30)
        .mean()
    )

    volume_std = (
        volume.rolling(30)
        .std()
    )

    x["volume_z"] = (
        volume
        - volume_mean
    ) / volume_std.replace(
        0,
        np.nan,
    )

    x["atr_pct"] = (
        (
            x["high"]
            - x["low"]
        )
        / close
    ) * 100

    return x


# ============================================================
# CREATE GPR MODEL
# ============================================================

def create_gpr_model():
    kernel = (
        ConstantKernel(
            1.0,
            (1e-3, 1e3),
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

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=42,
    )


# ============================================================
# TRAIN CURRENT GPR
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

    data["future_return"] = (
        data["close"].shift(
            -horizon_bars
        )
        / data["close"]
        - 1
    ) * 100

    model_data = (
        data
        .dropna(
            subset=FEATURES
            + ["future_return"]
        )
        .copy()
        .tail(180)
    )

    if len(model_data) < 70:

        raise RuntimeError(
            "Not enough valid training "
            f"data: {len(model_data)}"
        )

    X = (
        model_data[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
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

    if len(X) < 70:

        raise RuntimeError(
            "Insufficient valid "
            "training rows"
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    model = create_gpr_model()

    model.fit(
        X_scaled,
        y.values,
    )

    latest = (
        data
        .dropna(
            subset=FEATURES
        )
        .iloc[-1]
    )

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

    prediction, std = (
        model.predict(
            X_latest_scaled,
            return_std=True,
        )
    )

    return (
        float(prediction[0]),
        float(std[0]),
        latest,
        data,
    )


# ============================================================
# HISTORICAL FORWARD VALIDATION
# ============================================================

def forward_validation(
    df,
    timeframe,
    horizon,
):
    """
    Walk-forward directional validation.

    Model hanya boleh menggunakan data sebelum
    titik prediksi.

    Jadi tidak memakai candle masa depan
    untuk membuat prediksi historis.
    """

    data = add_features(
        df,
        timeframe,
    ).copy()

    validation = {}

    for forward_bars in FORWARD_HORIZONS:

        data["target_forward"] = (
            data["close"].shift(
                -forward_bars
            )
            / data["close"]
            - 1
        ) * 100

        valid_data = (
            data
            .dropna(
                subset=FEATURES
                + ["target_forward"]
            )
            .copy()
        )

        if len(valid_data) < 100:

            validation[
                forward_bars
            ] = {
                "hit_rate": 0.0,
                "avg_return": 0.0,
                "median_return": 0.0,
                "samples": 0,
            }

            continue

        # Ambil beberapa titik terakhir.
        # Ini menjaga scanner tetap cukup cepat.
        test_count = min(
            VALIDATION_POINTS,
            max(
                4,
                len(valid_data) - 70,
            ),
        )

        test_start = (
            len(valid_data)
            - test_count
        )

        predictions = []
        actuals = []

        for i in range(
            test_start,
            len(valid_data),
        ):

            train_part = valid_data.iloc[
                :i
            ]

            test_row = valid_data.iloc[
                i
            ]

            if len(train_part) < 70:
                continue

            X_train = (
                train_part[FEATURES]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            y_train = train_part[
                "target_forward"
            ]

            valid_train = (
                X_train.notna()
                .all(axis=1)
                & y_train.notna()
            )

            X_train = X_train.loc[
                valid_train
            ]

            y_train = y_train.loc[
                valid_train
            ]

            if len(X_train) < 70:
                continue

            scaler = StandardScaler()

            X_scaled = (
                scaler.fit_transform(
                    X_train
                )
            )

            model = create_gpr_model()

            model.fit(
                X_scaled,
                y_train.values,
            )

            X_test = (
                test_row[FEATURES]
                .values
                .reshape(1, -1)
            )

            X_test_scaled = (
                scaler.transform(
                    X_test
                )
            )

            pred = model.predict(
                X_test_scaled
            )[0]

            actual = float(
                test_row[
                    "target_forward"
                ]
            )

            predictions.append(
                float(pred)
            )

            actuals.append(
                actual
            )

        if not predictions:

            validation[
                forward_bars
            ] = {
                "hit_rate": 0.0,
                "avg_return": 0.0,
                "median_return": 0.0,
                "samples": 0,
            }

            continue

        pred_arr = np.array(
            predictions
        )

        actual_arr = np.array(
            actuals
        )

        correct = (
            np.sign(pred_arr)
            == np.sign(actual_arr)
        )

        hit_rate = float(
            correct.mean()
        )

        avg_return = float(
            actual_arr.mean()
        )

        median_return = float(
            np.median(actual_arr)
        )

        validation[
            forward_bars
        ] = {
            "hit_rate": hit_rate,
            "avg_return": avg_return,
            "median_return": median_return,
            "samples": len(actual_arr),
        }

    return validation


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    prediction,
    uncertainty,
    latest,
):
    ratio, probability, label = (
        calculate_confidence(
            prediction,
            uncertainty,
        )
    )

    magnitude = float(
        np.clip(
            abs(prediction) / 5.0,
            0,
            1,
        )
    )

    confidence_component = float(
        np.clip(
            (ratio - 0.5) / 2.5,
            0,
            1,
        )
    )

    rsi = float(
        latest["rsi14"]
    )

    if prediction > 0:

        if 45 <= rsi <= 70:
            rsi_component = 1.0

        elif (
            35 <= rsi < 45
            or 70 < rsi <= 78
        ):
            rsi_component = 0.75

        else:
            rsi_component = 0.40

    else:

        if 30 <= rsi <= 55:
            rsi_component = 1.0

        elif (
            22 <= rsi < 30
            or 55 < rsi <= 65
        ):
            rsi_component = 0.75

        else:
            rsi_component = 0.40

    momentum_values = [
        float(latest["mom_6"]),
        float(latest["mom_12"]),
        float(latest["mom_24"]),
    ]

    avg_momentum = float(
        np.nanmean(
            momentum_values
        )
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

    score = (
        magnitude * 35
        + confidence_component * 30
        + rsi_component * 15
        + momentum_alignment * 20
    )

    score = float(
        np.clip(
            score,
            0,
            100,
        )
    )

    return (
        score,
        ratio,
        probability,
        label,
    )


# ============================================================
# FINAL TRADING SCORE
# ============================================================

def calculate_final_score(
    gpr_score,
    confidence_ratio,
    agreement,
    validation_5,
    validation_10,
):
    """
    Final score khusus ranking.

    GPR tetap menjadi komponen terbesar.
    Forward validation digunakan sebagai filter
    kualitas historis, bukan sebagai pengganti GPR.
    """

    gpr_component = (
        np.clip(
            gpr_score / 100.0,
            0,
            1,
        )
    )

    confidence_component = np.clip(
        confidence_ratio / 3.0,
        0,
        1,
    )

    agreement_component = (
        np.clip(
            agreement / 100.0,
            0,
            1,
        )
    )

    validation_5_component = np.clip(
        validation_5,
        0,
        1,
    )

    validation_10_component = np.clip(
        validation_10,
        0,
        1,
    )

    final_score = 100 * (
        0.40 * gpr_component
        + 0.15 * confidence_component
        + 0.10 * agreement_component
        + 0.15 * validation_5_component
        + 0.20 * validation_10_component
    )

    return float(
        np.clip(
            final_score,
            0,
            100,
        )
    )


# ============================================================
# TRADE STATUS
# ============================================================

def trade_status(
    final_score,
    confidence_ratio,
    validation_5,
    validation_10,
):
    if (
        final_score >= 80
        and confidence_ratio >= 1.5
        and validation_5 >= 0.60
        and validation_10 >= 0.60
    ):
        return "TRADE"

    if (
        final_score >= 70
        and confidence_ratio >= 1.0
        and validation_5 >= 0.50
    ):
        return "WATCH"

    return "AVOID"


# ============================================================
# SIGNAL AGREEMENT
# ============================================================

def calculate_agreement(
    prediction,
    latest,
):
    components = np.array(
        [
            np.sign(
                latest["mom_6"]
            ),
            np.sign(
                latest["mom_12"]
            ),
            np.sign(
                latest["mom_24"]
            ),
            np.sign(
                latest["ema12_24_spread"]
            ),
        ]
    )

    target = (
        1
        if prediction >= 0
        else -1
    )

    valid = components != 0

    if not valid.any():
        return 0.0

    return float(
        (
            components[valid]
            == target
        ).mean()
        * 100
    )


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_symbol(
    symbol,
    timeframe,
    horizon_bars,
):
    bar = TIMEFRAME_CONFIG[
        timeframe
    ]["bar"]

    df = get_okx_candles(
        symbol,
        bar=bar,
        limit=260,
    )

    (
        prediction,
        uncertainty,
        latest,
        feature_data,
    ) = train_gpr(
        df,
        timeframe,
        horizon_bars,
    )

    (
        gpr_score,
        confidence_ratio,
        conf_probability,
        conf_label,
    ) = calculate_score(
        prediction,
        uncertainty,
        latest,
    )

    signal = (
        "LONG"
        if prediction >= 0
        else "SHORT"
    )

    agreement = calculate_agreement(
        prediction,
        latest,
    )

    # --------------------------------------------------------
    # FORWARD VALIDATION
    # --------------------------------------------------------

    validation = forward_validation(
        df,
        timeframe,
        horizon_bars,
    )

    validation_5 = validation[
        5
    ]

    validation_10 = validation[
        10
    ]

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_score = calculate_final_score(
        gpr_score=gpr_score,
        confidence_ratio=confidence_ratio,
        agreement=agreement,
        validation_5=validation_5[
            "hit_rate"
        ],
        validation_10=validation_10[
            "hit_rate"
        ],
    )

    status = trade_status(
        final_score,
        confidence_ratio,
        validation_5[
            "hit_rate"
        ],
        validation_10[
            "hit_rate"
        ],
    )

    # --------------------------------------------------------
    # FORWARD EXPECTED RETURN
    # --------------------------------------------------------

    direction = (
        1
        if signal == "LONG"
        else -1
    )

    avg_forward_5 = (
        direction
        * validation_5[
            "avg_return"
        ]
    )

    avg_forward_10 = (
        direction
        * validation_10[
            "avg_return"
        ]
    )

    return {
        "COIN": symbol,
        "SIGNAL": signal,

        "GPR SCORE": gpr_score,
        "FINAL SCORE": final_score,

        "GPR RETURN": prediction,
        "UNCERTAINTY": uncertainty,

        "CONFIDENCE": confidence_ratio,
        "CONF %": conf_probability * 100,
        "CONF LABEL": conf_label,

        "AGREEMENT": agreement,

        "5C HIT RATE": validation_5[
            "hit_rate"
        ] * 100,

        "10C HIT RATE": validation_10[
            "hit_rate"
        ] * 100,

        "5C AVG RETURN": avg_forward_5,
        "10C AVG RETURN": avg_forward_10,

        "5C SAMPLES": validation_5[
            "samples"
        ],

        "10C SAMPLES": validation_10[
            "samples"
        ],

        "STATUS": status,

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
):
    results = []
    errors = []

    progress = st.progress(0)
    status = st.empty()

    total = len(symbols)

    for i, symbol in enumerate(
        symbols
    ):

        status.write(
            f"Scanning {symbol} "
            f"({i + 1}/{total})..."
        )

        try:

            results.append(
                analyze_symbol(
                    symbol,
                    timeframe,
                    horizon_bars,
                )
            )

        except Exception as exc:

            errors.append(
                {
                    "Symbol": symbol,
                    "Status": str(exc),
                }
            )

        progress.progress(
            (i + 1) / total
        )

    progress.empty()
    status.empty()

    return results, errors


# ============================================================
# DISPLAY HELPERS
# ============================================================

def metric_card(
    title,
    value,
    subtitle="",
):
    with st.container(
        border=True
    ):

        st.markdown(
            f"**{title}**"
        )

        st.markdown(
            f"### {value}"
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

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["15m", "1H"],
    index=0,
)

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
        format_horizon(
            x,
            timeframe,
        ),
)

coin_count = st.sidebar.selectbox(
    "Jumlah coin",
    [20, 30, 50, 79],
    index=3,
)

minimum_score = st.sidebar.slider(
    "Minimum Final Score",
    min_value=50,
    max_value=95,
    value=70,
    step=1,
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence Ratio",
    min_value=0.5,
    max_value=4.0,
    value=1.5,
    step=0.1,
)

min_validation = st.sidebar.slider(
    "Minimum 10C Hit Rate",
    min_value=40,
    max_value=90,
    value=60,
    step=5,
)

st.sidebar.caption(
    "Confidence Ratio = "
    "|Forecast| ÷ Uncertainty"
)

st.sidebar.caption(
    "Final Score memasukkan "
    "validasi forward 5C + 10C."
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
# CLEAR CACHE
# ============================================================

if clear_button:

    st.cache_data.clear()

    for key in [
        "results",
        "errors",
        "timeframe",
        "horizon",
        "elapsed",
    ]:

        st.session_state.pop(
            key,
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
    f"Target: USDT Perpetual · "
    f"{timeframe} · "
    f"Horizon "
    f"{format_horizon(horizon, timeframe)} · "
    "GPR sebagai mesin ranking"
)

st.info(
    "Public market data OKX tanpa API key. "
    "Universe futures menggunakan daftar static "
    "agar scanner tidak bergantung pada endpoint "
    "instruments/exchangeInfo."
)


# ============================================================
# RUN SCAN
# ============================================================

if scan_button:

    selected_symbols = (
        UNIVERSE[:coin_count]
    )

    start_time = time.time()

    with st.spinner(
        f"Menjalankan GPR Futures "
        f"Scanner {timeframe}..."
    ):

        results, errors = run_scan(
            selected_symbols,
            timeframe,
            horizon,
        )

    elapsed = (
        time.time()
        - start_time
    )

    if results:

        df_result = pd.DataFrame(
            results
        )

        # ----------------------------------------------------
        # FINAL RANKING
        # ----------------------------------------------------

        df_result = (
            df_result
            .sort_values(
                [
                    "FINAL SCORE",
                    "10C HIT RATE",
                    "5C HIT RATE",
                    "GPR SCORE",
                    "CONFIDENCE",
                    "COIN",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
                kind="mergesort",
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
            "Tidak ada coin yang berhasil "
            "dianalisis."
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
# SHOW RESULTS
# ============================================================

if "results" not in st.session_state:

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
        - Model: **Gaussian Process Regression**
        - Signal: **LONG / SHORT**
        - Ranking utama: **FINAL SCORE**
        - Validasi: **5 candle + 10 candle**
        - Confidence: **Forecast ÷ Uncertainty**
        - Forward validation: **walk-forward**
        """
    )

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
    # FILTER CANDIDATES
    # ========================================================

    candidates = df[
        (
            df["FINAL SCORE"]
            >= minimum_score
        )
        & (
            df["CONFIDENCE"]
            >= min_confidence
        )
        & (
            df["10C HIT RATE"]
            >= min_validation
        )
    ].copy()

    st.success(
        f"Scan selesai: "
        f"{len(df)} berhasil, "
        f"{len(errors_df)} gagal "
        f"dalam {elapsed:.1f} detik."
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "UNIVERSE",
            len(df),
            "USDT Perpetual",
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
            format_horizon(
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


    # ========================================================
    # SECOND SUMMARY
    # ========================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "BERHASIL DIANALISIS",
            len(df),
        )

    with c2:

        score_pass = int(
            (
                df["FINAL SCORE"]
                >= minimum_score
            ).sum()
        )

        metric_card(
            "LOLOS FINAL SCORE",
            score_pass,
            f"≥ {minimum_score}",
        )

    with c3:

        metric_card(
            "KANDIDAT TRADING",
            len(candidates),
            (
                f"Score ≥ {minimum_score} · "
                f"10C ≥ {min_validation}%"
            ),
        )

    with c4:

        top = (
            candidates.iloc[0]
            if not candidates.empty
            else df.iloc[0]
        )

        metric_card(
            "TOP FINAL",
            top["COIN"],
            (
                f"{top['SIGNAL']} · "
                f"Final "
                f"{top['FINAL SCORE']:.1f}"
            ),
        )


    # ========================================================
    # VALIDATION GUIDE
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Forward Validation 5–10 Candle"
    )

    st.markdown(
        """
        <div class="guide">

        <b>5C HIT RATE</b> = persentase historis
        ketika arah prediksi model sesuai dengan
        arah pergerakan aktual 5 candle berikutnya.

        <br><br>

        <b>10C HIT RATE</b> = persentase historis
        ketika arah prediksi model sesuai dengan
        arah pergerakan aktual 10 candle berikutnya.

        <br><br>

        Untuk timeframe <b>15m</b>:

        <br>

        • 5 candle = <b>75 menit</b>

        <br>

        • 10 candle = <b>150 menit / 2,5 jam</b>

        <br><br>

        <b>Catatan penting:</b>
        Forward validation adalah validasi historis
        walk-forward. Angka tersebut bukan jaminan
        bahwa trade berikutnya akan profit.

        </div>
        """,
        unsafe_allow_html=True,
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
            "STATUS",
            "FINAL SCORE",
            "GPR SCORE",
            "GPR RETURN",
            "UNCERTAINTY",
            "CONFIDENCE",
            "5C HIT RATE",
            "10C HIT RATE",
            "5C AVG RETURN",
            "10C AVG RETURN",
            "PRICE",
            "RSI",
            "MOM 6H",
            "MOM 12H",
            "MOM 24H",
            "VOL Z",
        ]
    ].copy()

    display_df[
        "FINAL SCORE"
    ] = display_df[
        "FINAL SCORE"
    ].round(1)

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
            f"{x:.2f}x"
    )

    for col in [
        "5C HIT RATE",
        "10C HIT RATE",
    ]:

        display_df[
            col
        ] = display_df[
            col
        ].map(
            lambda x:
                f"{x:.1f}%"
        )

    for col in [
        "5C AVG RETURN",
        "10C AVG RETURN",
    ]:

        display_df[
            col
        ] = display_df[
            col
        ].map(
            lambda x:
                f"{x:+.2f}%"
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
        "MOM 24H",
    ]:

        display_df[
            col
        ] = display_df[
            col
        ].map(
            lambda x:
                f"{x:+.2f}%"
        )

    display_df[
        "VOL Z"
    ] = display_df[
        "VOL Z"
    ].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(
                "🏆 Rank",
                width="small",
            ),

            "COIN": st.column_config.TextColumn(
                "COIN"
            ),

            "SIGNAL": st.column_config.TextColumn(
                "SIGNAL"
            ),

            "STATUS": st.column_config.TextColumn(
                "STATUS"
            ),

            "FINAL SCORE": st.column_config.NumberColumn(
                "FINAL SCORE"
            ),

            "GPR SCORE": st.column_config.NumberColumn(
                "GPR SCORE"
            ),

            "GPR RETURN": st.column_config.TextColumn(
                "GPR RETURN"
            ),

            "UNCERTAINTY": st.column_config.TextColumn(
                "UNCERTAINTY"
            ),

            "CONFIDENCE": st.column_config.TextColumn(
                "CONFIDENCE RATIO"
            ),

            "5C HIT RATE": st.column_config.TextColumn(
                "5C HIT RATE"
            ),

            "10C HIT RATE": st.column_config.TextColumn(
                "10C HIT RATE"
            ),

            "5C AVG RETURN": st.column_config.TextColumn(
                "5C AVG RETURN"
            ),

            "10C AVG RETURN": st.column_config.TextColumn(
                "10C AVG RETURN"
            ),
        },
    )


    # ========================================================
    # TOP SIGNALS
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Top Trading Candidates"
    )

    top_df = (
        candidates.head(6)
        if not candidates.empty
        else df.head(6)
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

                signal_text = (
                    "🟢 LONG"
                    if row["SIGNAL"]
                    == "LONG"
                    else "🔴 SHORT"
                )

                status_text = (
                    "🟢 TRADE"
                    if row["STATUS"]
                    == "TRADE"
                    else (
                        "🟡 WATCH"
                        if row["STATUS"]
                        == "WATCH"
                        else "🔴 AVOID"
                    )
                )

                st.markdown(
                    f"### "
                    f"{row['COIN']} · "
                    f"{signal_text}"
                )

                st.caption(
                    f"Status: {status_text}"
                )

                a, b, c = st.columns(3)

                with a:

                    st.metric(
                        "Final Score",
                        f"{row['FINAL SCORE']:.1f}",
                    )

                with b:

                    st.metric(
                        "GPR Score",
                        f"{row['GPR SCORE']:.1f}",
                    )

                with c:

                    st.metric(
                        "Confidence",
                        f"{row['CONFIDENCE']:.2f}x",
                    )

                a, b = st.columns(2)

                with a:

                    st.metric(
                        "5C Hit Rate",
                        f"{row['5C HIT RATE']:.1f}%",
                    )

                with b:

                    st.metric(
                        "10C Hit Rate",
                        f"{row['10C HIT RATE']:.1f}%",
                    )

                st.caption(
                    f"5C Avg: "
                    f"{row['5C AVG RETURN']:+.2f}% · "
                    f"10C Avg: "
                    f"{row['10C AVG RETURN']:+.2f}%"
                )


    # ========================================================
    # DETAIL
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 Detail Analysis"
    )

    selected_coin = st.selectbox(
        "Pilih coin",
        df["COIN"].tolist(),
    )

    selected = df[
        df["COIN"]
        == selected_coin
    ].iloc[0]


    # --------------------------------------------------------
    # SIGNAL HEADER
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "SIGNAL",
            selected["SIGNAL"],
        )

    with c2:

        st.metric(
            "FINAL SCORE",
            f"{selected['FINAL SCORE']:.1f}",
        )

    with c3:

        st.metric(
            "GPR SCORE",
            f"{selected['GPR SCORE']:.1f}",
        )

    with c4:

        st.metric(
            "STATUS",
            selected["STATUS"],
        )


    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "GPR FORECAST",
            f"{selected['GPR RETURN']:+.2f}%",
        )

    with c2:

        st.metric(
            "UNCERTAINTY",
            f"{selected['UNCERTAINTY']:.2f}%",
        )

    with c3:

        st.metric(
            "CONFIDENCE RATIO",
            f"{selected['CONFIDENCE']:.2f}x",
        )

    with c4:

        st.metric(
            "CONF %",
            f"{selected['CONF %']:.1f}%",
        )


    # --------------------------------------------------------
    # FORWARD VALIDATION DETAIL
    # --------------------------------------------------------

    st.markdown(
        "#### 📊 Forward Validation"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "5C HIT RATE",
            f"{selected['5C HIT RATE']:.1f}%",
        )

    with c2:

        st.metric(
            "10C HIT RATE",
            f"{selected['10C HIT RATE']:.1f}%",
        )

    with c3:

        st.metric(
            "5C AVG RETURN",
            f"{selected['5C AVG RETURN']:+.2f}%",
        )

    with c4:

        st.metric(
            "10C AVG RETURN",
            f"{selected['10C AVG RETURN']:+.2f}%",
        )


    st.caption(
        f"Validation samples: "
        f"5C = {int(selected['5C SAMPLES'])}, "
        f"10C = {int(selected['10C SAMPLES'])}"
    )


    # --------------------------------------------------------
    # CONFIDENCE GUIDE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📊 Cara Membaca Confidence"
    )

    st.markdown(
        """
        <div class="guide">

        <b>Confidence Ratio =
        |Forecast| ÷ Uncertainty</b>

        <br><br>

        <b>&lt; 0.50</b> 🔴 Sangat Lemah

        &nbsp; | &nbsp;

        <b>0.50–0.99</b> 🟠 Lemah

        &nbsp; | &nbsp;

        <b>1.00–1.49</b> 🟡 Sedang

        &nbsp; | &nbsp;

        <b>1.50–1.99</b> 🟢 Baik

        &nbsp; | &nbsp;

        <b>2.00–2.49</b> 🟢 Kuat

        &nbsp; | &nbsp;

        <b>2.50–2.99</b> 🟢 Sangat Kuat

        &nbsp; | &nbsp;

        <b>≥ 3.00</b> 🔵 Ekstrem

        <br><br>

        Confidence Ratio bukan probabilitas profit.
        Untuk keputusan ranking, kita sekarang juga
        melihat validasi 5C dan 10C.

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # TECHNICAL DATA
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📐 Technical Data"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Harga",
            f"{selected['PRICE']:.8g}",
        )

    with c2:

        st.metric(
            "RSI 14",
            f"{selected['RSI']:.1f}",
        )

    with c3:

        st.metric(
            "Momentum 6H",
            f"{selected['MOM 6H']:+.2f}%",
        )

    with c4:

        st.metric(
            "Momentum 12H",
            f"{selected['MOM 12H']:+.2f}%",
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Momentum 24H",
            f"{selected['MOM 24H']:+.2f}%",
        )

    with c2:

        st.metric(
            "Volume Z",
            f"{selected['VOL Z']:.2f}",
        )

    with c3:

        st.metric(
            "EMA12 Distance",
            f"{selected['EMA12 DIST']:+.2f}%",
        )

    with c4:

        st.metric(
            "EMA24 Distance",
            f"{selected['EMA24 DIST']:+.2f}%",
        )

    st.caption(
        f"Last confirmed candle: "
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
        mime="text/csv",
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
