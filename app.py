import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="GPR Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CACHE_SECONDS = 300
REQUEST_TIMEOUT = 15

# Target utama
TIMEFRAME = "1h"
BYBIT_INTERVAL = "60"
BINANCE_INTERVAL = "1h"

# Minimal candle untuk GPR
MIN_CANDLES = 100


# ============================================================
# HTTP SESSION
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) "
        "AppleWebKit/605.1.15 Version/18.5 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "application/json",
})


# ============================================================
# BYBIT
# ============================================================

BYBIT_BASE = "https://api.bybit.com"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_bybit_symbols():
    """
    Ambil seluruh USDT Perpetual aktif dari Bybit.
    Menggunakan pagination cursor karena linear symbols > 500.
    """

    symbols = []
    cursor = None

    for _ in range(10):

        params = {
            "category": "linear",
            "status": "Trading",
            "limit": 1000,
        }

        if cursor:
            params["cursor"] = cursor

        r = SESSION.get(
            f"{BYBIT_BASE}/v5/market/instruments-info",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        r.raise_for_status()

        data = r.json()

        if data.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit instruments error: {data.get('retMsg')}"
            )

        result = data.get("result", {})

        for item in result.get("list", []):

            symbol = item.get("symbol", "")
            contract_type = item.get("contractType", "")
            quote_coin = item.get("quoteCoin", "")
            status = item.get("status", "")

            if (
                status == "Trading"
                and quote_coin == "USDT"
                and contract_type == "LinearPerpetual"
                and symbol.endswith("USDT")
            ):
                symbols.append(symbol)

        cursor = result.get("nextPageCursor")

        if not cursor:
            break

    symbols = sorted(set(symbols))

    if len(symbols) < 50:
        raise RuntimeError(
            f"Bybit hanya mengembalikan {len(symbols)} perpetual."
        )

    return symbols


def get_bybit_klines(symbol, limit=250):

    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": BYBIT_INTERVAL,
        "limit": limit,
    }

    r = SESSION.get(
        f"{BYBIT_BASE}/v5/market/kline",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    data = r.json()

    if data.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit kline {symbol}: {data.get('retMsg')}"
        )

    rows = data.get("result", {}).get("list", [])

    if not rows:
        raise RuntimeError(f"Bybit tidak punya candle untuk {symbol}")

    # Bybit:
    # [timestamp, open, high, low, close, volume, turnover]
    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]),
        unit="ms",
        utc=True,
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("time")
    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df.reset_index(drop=True)


# ============================================================
# BINANCE FUTURES FALLBACK
# ============================================================

BINANCE_BASE = "https://fapi.binance.com"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def get_binance_symbols():

    r = SESSION.get(
        f"{BINANCE_BASE}/fapi/v1/exchangeInfo",
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    data = r.json()

    symbols = []

    for item in data.get("symbols", []):

        if (
            item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
        ):
            symbol = item.get("symbol")

            if symbol:
                symbols.append(symbol)

    symbols = sorted(set(symbols))

    if len(symbols) < 50:
        raise RuntimeError(
            f"Binance hanya mengembalikan {len(symbols)} perpetual."
        )

    return symbols


def get_binance_klines(symbol, limit=250):

    params = {
        "symbol": symbol,
        "interval": BINANCE_INTERVAL,
        "limit": min(limit, 1000),
    }

    r = SESSION.get(
        f"{BINANCE_BASE}/fapi/v1/klines",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    r.raise_for_status()

    rows = r.json()

    if not isinstance(rows, list) or not rows:
        raise RuntimeError(
            f"Binance tidak punya candle untuk {symbol}"
        )

    # Binance:
    # open time
    # open
    # high
    # low
    # close
    # volume
    # ...

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]),
        unit="ms",
        utc=True,
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

    df = df.sort_values("time")

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    return df[
        [
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].reset_index(drop=True)


# ============================================================
# PROVIDER ROUTER
# ============================================================

def get_symbols():

    errors = []

    # Primary
    try:
        symbols = get_bybit_symbols()

        return symbols, "BYBIT"

    except Exception as exc:

        errors.append(
            f"Bybit: {type(exc).__name__}: {str(exc)[:100]}"
        )

    # Fallback
    try:
        symbols = get_binance_symbols()

        return symbols, "BINANCE"

    except Exception as exc:

        errors.append(
            f"Binance: {type(exc).__name__}: {str(exc)[:100]}"
        )

    raise RuntimeError(
        "Semua provider gagal mengambil universe.\n"
        + "\n".join(errors)
    )


def get_candles(symbol, provider, limit=250):

    errors = []

    if provider == "BYBIT":

        try:
            return get_bybit_klines(symbol, limit)

        except Exception as exc:

            errors.append(
                f"Bybit: {type(exc).__name__}: {str(exc)[:100]}"
            )

            # Coba Binance untuk simbol yang sama
            try:

                return get_binance_klines(
                    symbol,
                    limit,
                )

            except Exception as exc2:

                errors.append(
                    f"Binance: {type(exc2).__name__}: {str(exc2)[:100]}"
                )

    elif provider == "BINANCE":

        try:
            return get_binance_klines(symbol, limit)

        except Exception as exc:

            errors.append(
                f"Binance: {type(exc).__name__}: {str(exc)[:100]}"
            )

    raise RuntimeError(
        f"{symbol}: tidak mendapatkan candle.\n"
        + " | ".join(errors)
    )


# ============================================================
# GPR CORE
# ============================================================

def ema(series, length):

    return series.ewm(
        span=length,
        adjust=False,
        min_periods=1,
    ).mean()


def matern32_kernel(
    x,
    z,
    amplitude,
    scale,
):

    d = x - z

    distance2 = (
        d[0] ** 2
        + 0.8 * d[1] ** 2
        + 0.6 * d[2] ** 2
        + 0.8 * d[3] ** 2
        + 0.6 * d[4] ** 2
    )

    u = (
        math.sqrt(
            3.0 * max(
                float(distance2),
                0.0,
            )
        )
        / scale
    )

    return (
        amplitude
        * (1.0 + u)
        * math.exp(-u)
    )


def pine_rma(values, length):

    values = np.asarray(
        values,
        dtype=float,
    )

    out = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    if (
        length <= 0
        or len(values) < length
    ):
        return out

    valid_idx = np.flatnonzero(
        np.isfinite(values)
    )

    if len(valid_idx) < length:
        return out

    seed_indices = valid_idx[:length]

    seed_end = int(
        seed_indices[-1]
    )

    out[seed_end] = float(
        np.mean(
            values[seed_indices]
        )
    )

    alpha = 1.0 / length

    for i in range(
        seed_end + 1,
        len(values),
    ):

        if np.isfinite(values[i]):

            prev = out[i - 1]

            if not np.isfinite(prev):

                prior = values[
                    max(0, i - length + 1):
                    i + 1
                ]

                if (
                    np.isfinite(prior).sum()
                    == length
                ):
                    out[i] = float(
                        np.mean(prior)
                    )

            else:

                out[i] = (
                    alpha * values[i]
                    + (1.0 - alpha) * prev
                )

    return out


# ============================================================
# GPR PARAMETERS
# ============================================================

def gpr_profile():

    # KHUSUS TARGET 1H
    return dict(
        training=35,
        momentum_len=5,
        vol_len=20,
        ema_fast=8,
        ema_slow=21,
        kernel_scale=1.6,
        noise_ratio=0.65,
        volume_fade=0.50,
    )


# ============================================================
# GPR MODEL
# ============================================================

def gpr_project(
    df,
    horizon=10,
    training=35,
    momentum_len=5,
    vol_len=20,
    ema_fast=8,
    ema_slow=21,
    kernel_scale=1.6,
    noise_ratio=0.65,
    volume_fade=0.50,
    band_sigma=1.96,
    validation_candles=3,
):

    min_needed = max(
        training
        + vol_len
        + ema_slow
        + 10,
        80,
    )

    if len(df) < min_needed + validation_candles:

        raise ValueError(
            f"Candle hanya {len(df)}; "
            f"minimal {min_needed + validation_candles}"
        )

    idx = (
        len(df)
        - 1
        - validation_candles
    )

    close = (
        df["close"]
        .astype(float)
        .to_numpy()
    )

    volume = (
        df["volume"]
        .fillna(0)
        .astype(float)
        .to_numpy()
    )

    high = (
        df["high"]
        .astype(float)
        .to_numpy()
    )

    low = (
        df["low"]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    logret = np.full(
        len(close),
        np.nan,
    )

    logret[1:] = np.log(
        np.maximum(
            close[1:],
            1e-12,
        )
        /
        np.maximum(
            close[:-1],
            1e-12,
        )
    )

    ret_series = pd.Series(logret)

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum = (
        ret_series
        .ewm(
            span=momentum_len,
            adjust=False,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    volatility = (
        ret_series
        .rolling(
            vol_len,
            min_periods=vol_len,
        )
        .std(ddof=0)
        .to_numpy()
    )

    vol_base = (
        pd.Series(volatility)
        .rolling(
            training,
            min_periods=training,
        )
        .mean()
        .to_numpy()
    )

    # --------------------------------------------------------
    # RELATIVE VOLUME
    # --------------------------------------------------------

    avg_volume = (
        pd.Series(volume)
        .rolling(
            vol_len,
            min_periods=vol_len,
        )
        .mean()
        .to_numpy()
    )

    rel_volume = np.zeros(
        len(close)
    )

    valid = (
        (volume > 0)
        & (avg_volume > 0)
        & np.isfinite(avg_volume)
    )

    rel_volume[valid] = np.log(
        volume[valid]
        /
        avg_volume[valid]
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    fast = ema(
        pd.Series(close),
        ema_fast,
    ).to_numpy()

    slow = ema(
        pd.Series(close),
        ema_slow,
    ).to_numpy()

    direction = (
        fast - slow
    ) / np.maximum(
        close,
        1e-12,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    prev_close = np.r_[
        np.nan,
        close[:-1],
    ]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(
                high - prev_close
            ),
            np.abs(
                low - prev_close
            ),
        ),
    )

    tr[0] = (
        high[0]
        - low[0]
    )

    atr = pine_rma(
        tr,
        vol_len,
    )

    vol_ref = max(
        float(
            vol_base[idx]
        )
        if np.isfinite(
            vol_base[idx]
        )
        else 0.0,
        0.0001,
    )

    if (
        not np.isfinite(
            volatility[idx]
        )
        or not np.isfinite(
            atr[idx]
        )
        or not np.isfinite(
            logret[idx]
        )
    ):

        raise ValueError(
            "Indikator belum siap."
        )

    # ========================================================
    # TRAINING SET
    # ========================================================

    X = []
    y = []

    for r in range(training):

        target_idx = (
            idx
            - training
            + 1
            + r
        )

        feature_idx = (
            target_idx - 1
        )

        if (
            feature_idx < 1
            or not np.isfinite(
                logret[target_idx]
            )
        ):
            continue

        X.append([
            logret[feature_idx]
            / vol_ref,

            momentum[feature_idx]
            / vol_ref,

            rel_volume[feature_idx],

            volatility[feature_idx]
            / vol_ref
            - 1.0,

            direction[feature_idx]
            / vol_ref,
        ])

        y.append(
            logret[target_idx]
        )

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    n = len(y)

    if n < 15:

        raise ValueError(
            "Training sample GPR terlalu sedikit."
        )

    # ========================================================
    # GAUSSIAN PROCESS
    # ========================================================

    signal_var = max(
        float(
            np.mean(y * y)
        ),
        1e-8,
    )

    noise_var = (
        signal_var
        * noise_ratio
        + 1e-10
    )

    K = np.empty(
        (n, n),
        dtype=float,
    )

    for i in range(n):

        for j in range(i + 1):

            value = matern32_kernel(
                X[i],
                X[j],
                signal_var,
                kernel_scale,
            )

            if i == j:
                value += noise_var

            K[i, j] = value
            K[j, i] = value

    K += (
        np.eye(n)
        * 1e-6
    )

    try:

        L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(
            L.T,
            np.linalg.solve(
                L,
                y,
            ),
        )

    except np.linalg.LinAlgError:

        K += (
            np.eye(n)
            * 1e-4
        )

        L = np.linalg.cholesky(K)

        alpha = np.linalg.solve(
            L.T,
            np.linalg.solve(
                L,
                y,
            ),
        )

    # ========================================================
    # FORECAST
    # ========================================================

    price_start = float(
        close[idx]
    )

    price_prev = price_start

    ret_prev = float(
        logret[idx]
    )

    mom_prev = float(
        momentum[idx]
    )

    fast_prev = float(
        fast[idx]
    )

    slow_prev = float(
        slow[idx]
    )

    volume_prev = float(
        rel_volume[idx]
    )

    vol_prev = float(
        volatility[idx]
    )

    cumulative_return = 0.0

    cumulative_variance = 0.0

    predictions = []

    total_steps = (
        horizon
        + validation_candles
    )

    for h in range(total_steps):

        fade = (
            volume_fade ** h
        )

        assumed_vol = (
            vol_ref
            + (
                vol_prev
                - vol_ref
            )
            * fade
        )

        x_star = np.array([

            ret_prev
            / vol_ref,

            mom_prev
            / vol_ref,

            volume_prev
            * fade,

            assumed_vol
            / vol_ref
            - 1.0,

            (
                (
                    fast_prev
                    - slow_prev
                )
                / max(
                    price_prev,
                    1e-12,
                )
            )
            / vol_ref,
        ])

        kstar = np.array([
            matern32_kernel(
                X[i],
                x_star,
                signal_var,
                kernel_scale,
            )
            for i in range(n)
        ])

        pred_ret = float(
            kstar @ alpha
        )

        solved = np.linalg.solve(
            L,
            kstar,
        )

        predictive_variance = max(
            float(
                signal_var
                + noise_var
                - solved @ solved
            ),
            1e-12,
        )

        cumulative_variance += (
            predictive_variance
        )

        cumulative_return += (
            pred_ret
        )

        price_pred = (
            price_start
            * math.exp(
                cumulative_return
            )
        )

        sigma_cum = math.sqrt(
            max(
                cumulative_variance,
                0.0,
            )
        )

        upper = (
            price_pred
            * math.exp(
                band_sigma
                * sigma_cum
            )
        )

        lower = (
            price_pred
            * math.exp(
                -band_sigma
                * sigma_cum
            )
        )

        real_idx = (
            idx
            + h
            + 1
        )

        actual_price = (
            float(
                close[real_idx]
            )
            if real_idx < len(df)
            else None
        )

        predictions.append({

            "step": h + 1,

            "is_future":
                actual_price is None,

            "price":
                price_pred,

            "actual_price":
                actual_price,

            "upper":
                upper,

            "lower":
                lower,

            "sigma_cum":
                sigma_cum,
        })

        price_prev = price_pred

        ret_prev = pred_ret

        a_m = (
            2.0
            /
            (momentum_len + 1.0)
        )

        mom_prev = (
            a_m * pred_ret
            +
            (1.0 - a_m)
            * mom_prev
        )

        a_f = (
            2.0
            /
            (ema_fast + 1.0)
        )

        a_s = (
            2.0
            /
            (ema_slow + 1.0)
        )

        fast_prev = (
            a_f * price_pred
            +
            (1.0 - a_f)
            * fast_prev
        )

        slow_prev = (
            a_s * price_pred
            +
            (1.0 - a_s)
            * slow_prev
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_errors = []
    hits = 0

    for p in predictions:

        if (
            not p["is_future"]
            and p["actual_price"]
            is not None
        ):

            err = (
                abs(
                    p["price"]
                    - p["actual_price"]
                )
                /
                max(
                    p["actual_price"],
                    1e-12,
                )
            )

            validation_errors.append(
                err
            )

            if (
                p["lower"]
                <= p["actual_price"]
                <= p["upper"]
            ):
                hits += 1

    val_count = len(
        validation_errors
    )

    validation_mape = (
        np.mean(
            validation_errors
        )
        if val_count
        else 1.0
    )

    validation_hit_rate = (
        hits / val_count
        if val_count
        else 0.0
    )

    # ========================================================
    # CURRENT STATE
    # ========================================================

    latest_price = float(
        close[-1]
    )

    trend = (
        "BULLISH"
        if fast[-1] > slow[-1]
        else "BEARISH"
    )

    mom_now = float(
        momentum[-1]
    )

    vol_now = float(
        volume[-1]
    )

    vol_avg_now = float(
        avg_volume[-1]
    ) if np.isfinite(
        avg_volume[-1]
    ) else 0.0

    relative_volume = (
        vol_now / vol_avg_now
        if vol_avg_now > 0
        else 0.0
    )

    final = predictions[-1]

    sigma_final = math.sqrt(
        max(
            cumulative_variance,
            0.0,
        )
    )

    uncertainty = (
        100.0
        * sigma_final
        /
        (
            abs(
                cumulative_return
            )
            + sigma_final
            + 1e-10
        )
    )

    future_return = (
        final["price"]
        /
        latest_price
        - 1.0
    ) * 100.0

    return {

        "price":
            latest_price,

        "trend":
            trend,

        "momentum":
            (
                "POSITIVE"
                if mom_now > 0
                else "NEGATIVE"
            ),

        "relative_volume":
            relative_volume,

        "gpr_return_pct":
            future_return,

        "uncertainty_pct":
            uncertainty,

        "projected_price":
            final["price"],

        "validation_mape":
            validation_mape,

        "validation_hit_rate":
            validation_hit_rate,

        "predictions":
            predictions,
    }


# ============================================================
# SCORING
# ============================================================

def calculate_score(result):

    hit_score = (
        result["validation_hit_rate"]
        * 30
    )

    mape_score = max(
        0,
        20
        -
        (
            result["validation_mape"]
            * 100
            * 4
        ),
    )

    trend_score = (
        15
        if result["trend"]
        == "BULLISH"
        else 0
    )

    momentum_score = (
        10
        if result["momentum"]
        == "POSITIVE"
        else 0
    )

    return_score = (
        25
        *
        max(
            0,
            min(
                result["gpr_return_pct"]
                / 5.0,
                1.0,
            ),
        )
    )

    score = (
        hit_score
        + mape_score
        + trend_score
        + momentum_score
        + return_score
    )

    return round(
        max(
            0,
            min(
                100,
                score,
            ),
        ),
        1,
    )


def make_signal(result):

    score = result["score"]

    ret = result[
        "gpr_return_pct"
    ]

    hit = result[
        "validation_hit_rate"
    ]

    trend = result[
        "trend"
    ]

    momentum = result[
        "momentum"
    ]

    # ========================================================
    # KANDIDAT LONG
    # ========================================================

    if (
        score >= 75
        and trend == "BULLISH"
        and momentum == "POSITIVE"
        and ret > 0
        and hit >= 0.66
    ):
        return "LONG CANDIDATE"

    # ========================================================
    # KANDIDAT SHORT
    # ========================================================

    if (
        score >= 75
        and trend == "BEARISH"
        and momentum == "NEGATIVE"
        and ret < 0
        and hit >= 0.66
    ):
        return "SHORT CANDIDATE"

    # ========================================================
    # WATCH
    # ========================================================

    if score >= 60:
        return "WATCH"

    return "AVOID"


# ============================================================
# ANALYZE
# ============================================================

def analyze_symbol(
    symbol,
    provider,
    horizon,
):

    df = get_candles(
        symbol,
        provider,
        limit=250,
    )

    if len(df) < MIN_CANDLES:

        raise ValueError(
            f"Candle {len(df)} < {MIN_CANDLES}"
        )

    params = gpr_profile()

    result = gpr_project(
        df,
        horizon=horizon,
        **params,
    )

    result["symbol"] = symbol
    result["provider"] = provider

    result["score"] = calculate_score(
        result
    )

    result["signal"] = make_signal(
        result
    )

    return result


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stMetric"] {
        border: 1px solid #26354a;
        padding: 12px 14px;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title(
    "📈 GPR CRYPTO FUTURES SCANNER"
)

st.caption(
    "USDT Perpetual · 1H · GPR sebagai mesin ranking utama"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader(
        "Scanner Settings"
    )

    horizon = st.selectbox(
        "Horizon GPR",
        [3, 5, 10, 20],
        index=2,
        format_func=lambda x:
            f"{x} candle / {x} jam",
    )

    max_assets = st.selectbox(
        "Jumlah coin",
        [50, 100, 200, 300, 500, 1000],
        index=1,
    )

    min_score = st.slider(
        "Minimum score",
        40,
        90,
        70,
        5,
    )

    workers = st.slider(
        "Parallel workers",
        2,
        12,
        6,
    )

    scan_button = st.button(
        "🚀 SCAN FUTURES 1H",
        type="primary",
        use_container_width=True,
    )

    clear_button = st.button(
        "🗑️ CLEAR CACHE",
        use_container_width=True,
    )


if clear_button:

    st.cache_data.clear()

    st.rerun()


# ============================================================
# UNIVERSE
# ============================================================

try:

    symbols, provider = get_symbols()

    st.success(
        f"Universe siap: {len(symbols)} USDT Perpetual · "
        f"provider utama: {provider}"
    )

except Exception as exc:

    st.error(
        f"Gagal mengambil universe futures: {exc}"
    )

    st.stop()


symbols = symbols[:max_assets]


# ============================================================
# SESSION STATE
# ============================================================

if "scan_results" not in st.session_state:

    st.session_state.scan_results = []

if "scan_errors" not in st.session_state:

    st.session_state.scan_errors = []


# ============================================================
# SCAN
# ============================================================

if scan_button:

    results = []
    errors = []

    progress = st.progress(
        0,
        text="Menyiapkan scanner..."
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
                provider,
                horizon,
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

            try:

                result = future.result()

                results.append(
                    result
                )

            except Exception as exc:

                errors.append({

                    "symbol":
                        symbol,

                    "error":
                        str(exc)[:250],
                })

            completed += 1

            progress.progress(
                completed / total,
                text=(
                    f"Scanning "
                    f"{completed}/{total} · "
                    f"{symbol}"
                ),
            )

            status.info(
                f"Berhasil: {len(results)} · "
                f"Gagal: {len(errors)}"
            )

    # Ranking
    results.sort(
        key=lambda x: (
            x["score"],
            x["gpr_return_pct"],
        ),
        reverse=True,
    )

    st.session_state.scan_results = (
        results
    )

    st.session_state.scan_errors = (
        errors
    )

    progress.progress(
        1.0,
        text="SCAN SELESAI"
    )

    st.success(
        f"Scan selesai: "
        f"{len(results)} berhasil · "
        f"{len(errors)} gagal"
    )


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.scan_results
)

errors = (
    st.session_state.scan_errors
)


if not results:

    st.info(
        "Belum ada hasil. "
        "Tekan SCAN FUTURES 1H."
    )

    st.stop()


qualified = [
    x
    for x in results
    if x["score"] >= min_score
]


longs = [
    x
    for x in qualified
    if x["signal"]
    == "LONG CANDIDATE"
]


shorts = [
    x
    for x in qualified
    if x["signal"]
    == "SHORT CANDIDATE"
]


best_long = (
    max(
        longs,
        key=lambda x:
        x["score"],
    )
    if longs
    else None
)


best_short = (
    max(
        shorts,
        key=lambda x:
        x["score"],
    )
    if shorts
    else None
)


# ============================================================
# METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Analyzed",
    len(results),
)

c2.metric(
    "Qualified",
    len(qualified),
)

c3.metric(
    "LONG Candidates",
    len(longs),
)

c4.metric(
    "SHORT Candidates",
    len(shorts),
)


# ============================================================
# TOP SIGNAL
# ============================================================

if best_long:

    st.success(
        f"🟢 TOP LONG: "
        f"{best_long['symbol']} · "
        f"Score {best_long['score']:.1f} · "
        f"GPR {best_long['gpr_return_pct']:+.2f}% · "
        f"Hit {best_long['validation_hit_rate']*100:.0f}%"
    )


if best_short:

    st.error(
        f"🔴 TOP SHORT: "
        f"{best_short['symbol']} · "
        f"Score {best_short['score']:.1f} · "
        f"GPR {best_short['gpr_return_pct']:+.2f}% · "
        f"Hit {best_short['validation_hit_rate']*100:.0f}%"
    )


# ============================================================
# TABLE
# ============================================================

st.subheader(
    "🏆 GPR Ranking"
)


table = []

for rank, item in enumerate(
    qualified,
    1,
):

    table.append({

        "Rank":
            rank,

        "Symbol":
            item["symbol"],

        "Score":
            item["score"],

        "Signal":
            item["signal"],

        "GPR Return":
            f"{item['gpr_return_pct']:+.2f}%",

        "T-3 Hit":
            f"{item['validation_hit_rate']*100:.0f}%",

        "MAPE":
            f"{item['validation_mape']*100:.2f}%",

        "Trend":
            item["trend"],

        "Momentum":
            item["momentum"],

        "Rel Volume":
            f"{item['relative_volume']:.2f}x",

        "Uncertainty":
            f"{item['uncertainty_pct']:.1f}%",

        "Price":
            item["price"],
    })


table_df = pd.DataFrame(
    table
)


st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# CSV
# ============================================================

st.download_button(
    "⬇️ Download GPR Ranking CSV",
    table_df.to_csv(
        index=False
    ).encode("utf-8"),
    file_name=(
        "gpr_crypto_futures_1h.csv"
    ),
    mime="text/csv",
)


# ============================================================
# DETAIL
# ============================================================

st.subheader(
    "📊 Detail GPR"
)

selected = st.selectbox(
    "Pilih coin",
    [
        x["symbol"]
        for x in qualified
    ]
    if qualified
    else [
        x["symbol"]
        for x in results
    ],
)


item = next(
    x
    for x in results
    if x["symbol"]
    == selected
)


d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Harga",
    f"{item['price']:,.6g}",
)

d2.metric(
    "GPR Return",
    f"{item['gpr_return_pct']:+.2f}%",
)

d3.metric(
    "Score",
    f"{item['score']:.1f}",
)

d4.metric(
    "Uncertainty",
    f"{item['uncertainty_pct']:.1f}%",
)


preds = item[
    "predictions"
]


chart = []

for p in preds:

    label = (
        f"+{p['step'] - 3}H"
        if p["is_future"]
        else f"T-{3 - p['step']}"
    )

    chart.append({

        "Candle":
            label,

        "GPR":
            p["price"],

        "Actual":
            p["actual_price"],

        "Upper":
            p["upper"],

        "Lower":
            p["lower"],
    })


chart_df = pd.DataFrame(
    chart
).set_index(
    "Candle"
)


st.line_chart(
    chart_df
)


# ============================================================
# ERRORS
# ============================================================

with st.expander(
    f"🧾 Error ({len(errors)})"
):

    if errors:

        error_df = pd.DataFrame(
            errors
        )

        st.dataframe(
            error_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "Tidak ada error."
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "GPR experimental scanner — bukan saran investasi. "
    "Data futures berasal dari public market data tanpa API trading."
)
