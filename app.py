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
    initial_sidebar_state="expanded",
)

CACHE_SECONDS = 300

OKX_BASE = "https://openapi.okx.com"
BITGET_BASE = "https://api.bitget.com"

# ============================================================
# STATIC FUTURES UNIVERSE
#
# Kita SENGAJA tidak mengambil universe dari exchange.
# Ini menghindari error 403/451 pada endpoint instruments/info.
#
# Format internal:
# BTC-USDT-SWAP
# ============================================================

FUTURES_SYMBOLS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "XRP-USDT-SWAP",
    "BNB-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "ADA-USDT-SWAP",
    "TRX-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "LINK-USDT-SWAP",
    "DOT-USDT-SWAP",
    "LTC-USDT-SWAP",
    "BCH-USDT-SWAP",
    "SUI-USDT-SWAP",
    "TON-USDT-SWAP",
    "NEAR-USDT-SWAP",
    "APT-USDT-SWAP",
    "ICP-USDT-SWAP",
    "ETC-USDT-SWAP",
    "XLM-USDT-SWAP",
    "HBAR-USDT-SWAP",
    "FIL-USDT-SWAP",
    "ATOM-USDT-SWAP",
    "ARB-USDT-SWAP",
    "OP-USDT-SWAP",
    "INJ-USDT-SWAP",
    "AAVE-USDT-SWAP",
    "UNI-USDT-SWAP",
    "MKR-USDT-SWAP",
    "PEPE-USDT-SWAP",
    "TAO-USDT-SWAP",
    "IMX-USDT-SWAP",
    "GRT-USDT-SWAP",
    "ALGO-USDT-SWAP",
    "VET-USDT-SWAP",
    "EOS-USDT-SWAP",
    "XTZ-USDT-SWAP",
    "RENDER-USDT-SWAP",
    "FET-USDT-SWAP",
    "SEI-USDT-SWAP",
    "TIA-USDT-SWAP",
    "WIF-USDT-SWAP",
    "BONK-USDT-SWAP",
    "JUP-USDT-SWAP",
    "ENA-USDT-SWAP",
    "WLD-USDT-SWAP",
    "ONDO-USDT-SWAP",
    "RUNE-USDT-SWAP",
    "MNT-USDT-SWAP",
    "STX-USDT-SWAP",
    "ORDI-USDT-SWAP",
    "THETA-USDT-SWAP",
    "SAND-USDT-SWAP",
    "MANA-USDT-SWAP",
    "AXS-USDT-SWAP",
    "GALA-USDT-SWAP",
    "APE-USDT-SWAP",
    "CRV-USDT-SWAP",
    "LDO-USDT-SWAP",
    "DYDX-USDT-SWAP",
    "COMP-USDT-SWAP",
    "SNX-USDT-SWAP",
    "SUSHI-USDT-SWAP",
    "1INCH-USDT-SWAP",
    "ZEC-USDT-SWAP",
    "DASH-USDT-SWAP",
    "XMR-USDT-SWAP",
    "ETC-USDT-SWAP",
    "KAS-USDT-SWAP",
    "JTO-USDT-SWAP",
    "PYTH-USDT-SWAP",
    "STRK-USDT-SWAP",
    "BLUR-USDT-SWAP",
    "MASK-USDT-SWAP",
    "ENS-USDT-SWAP",
    "EGLD-USDT-SWAP",
    "FLOW-USDT-SWAP",
    "KSM-USDT-SWAP",
    "ZK-USDT-SWAP",
    "NOT-USDT-SWAP",
]

FUTURES_SYMBOLS = list(dict.fromkeys(FUTURES_SYMBOLS))


# ============================================================
# HELPERS
# ============================================================

def display_code(inst_id):
    """
    BTC-USDT-SWAP -> BTC
    """
    return inst_id.replace("-USDT-SWAP", "")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_http_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/155.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Connection": "keep-alive",
    })

    return session


# ============================================================
# OKX
# ============================================================

def get_okx_candles(inst_id, limit=300):
    """
    OKX public market candles.

    Endpoint:
    /api/v5/market/candles

    1H bar:
    bar=1H

    Kita hanya mengambil candle CLOSED.
    """

    session = get_http_session()

    url = f"{OKX_BASE}/api/v5/market/candles"

    params = {
        "instId": inst_id,
        "bar": "1H",
        "limit": str(min(limit, 300)),
    }

    last_error = None

    for attempt in range(3):
        try:
            response = session.get(
                url,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:150]}"
                )

            payload = response.json()

            if str(payload.get("code")) != "0":
                raise RuntimeError(
                    f"OKX code={payload.get('code')} "
                    f"msg={payload.get('msg')}"
                )

            rows = payload.get("data", [])

            if not rows:
                raise RuntimeError("OKX mengembalikan candle kosong.")

            records = []

            for row in rows:
                if len(row) < 7:
                    continue

                records.append({
                    "time": pd.to_datetime(
                        int(row[0]),
                        unit="ms",
                        utc=True,
                    ),
                    "open": safe_float(row[1]),
                    "high": safe_float(row[2]),
                    "low": safe_float(row[3]),
                    "close": safe_float(row[4]),
                    "volume": safe_float(row[5], 0.0),
                })

            df = pd.DataFrame(records)

            if df.empty:
                raise RuntimeError("Data candle OKX kosong setelah parsing.")

            df = df.sort_values("time").reset_index(drop=True)

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce",
                )

            df = df.dropna(
                subset=["open", "high", "low", "close"]
            )

            # ====================================================
            # PENTING:
            # Buang candle 1H yang masih berjalan.
            # ====================================================

            now = pd.Timestamp.now(tz="UTC")

            if len(df) > 0:
                last_time = df.iloc[-1]["time"]

                if last_time + pd.Timedelta(hours=1) > now:
                    df = df.iloc[:-1].copy()

            if len(df) < 80:
                raise RuntimeError(
                    f"OKX hanya memiliki {len(df)} candle closed."
                )

            return df.reset_index(drop=True)

        except Exception as exc:
            last_error = exc

            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))

    raise RuntimeError(
        f"OKX gagal: {type(last_error).__name__}: {str(last_error)[:180]}"
    )


# ============================================================
# BITGET FALLBACK
# ============================================================

def convert_okx_to_bitget_symbol(inst_id):
    """
    BTC-USDT-SWAP -> BTCUSDT
    """

    return inst_id.replace("-USDT-SWAP", "USDT")


def get_bitget_candles(inst_id, limit=300):
    """
    Bitget USDT-FUTURES fallback.

    Endpoint:
    /api/v2/mix/market/candles

    granularity=1H
    """

    session = get_http_session()

    symbol = convert_okx_to_bitget_symbol(inst_id)

    url = f"{BITGET_BASE}/api/v2/mix/market/candles"

    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES",
        "granularity": "1H",
        "limit": str(min(limit, 200)),
    }

    last_error = None

    for attempt in range(3):

        try:

            response = session.get(
                url,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:150]}"
                )

            payload = response.json()

            if str(payload.get("code")) != "00000":
                raise RuntimeError(
                    f"Bitget code={payload.get('code')} "
                    f"msg={payload.get('msg')}"
                )

            rows = payload.get("data", [])

            if not rows:
                raise RuntimeError(
                    "Bitget mengembalikan candle kosong."
                )

            records = []

            for row in rows:

                if len(row) < 7:
                    continue

                records.append({
                    "time": pd.to_datetime(
                        int(row[0]),
                        unit="ms",
                        utc=True,
                    ),
                    "open": safe_float(row[1]),
                    "high": safe_float(row[2]),
                    "low": safe_float(row[3]),
                    "close": safe_float(row[4]),
                    "volume": safe_float(row[5], 0.0),
                })

            df = pd.DataFrame(records)

            if df.empty:
                raise RuntimeError(
                    "Data candle Bitget kosong setelah parsing."
                )

            df = df.sort_values("time").reset_index(drop=True)

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

            df = df.dropna(
                subset=["open", "high", "low", "close"]
            )

            now = pd.Timestamp.now(tz="UTC")

            if len(df) > 0:

                last_time = df.iloc[-1]["time"]

                if last_time + pd.Timedelta(hours=1) > now:
                    df = df.iloc[:-1].copy()

            if len(df) < 80:
                raise RuntimeError(
                    f"Bitget hanya memiliki {len(df)} candle closed."
                )

            return df.reset_index(drop=True)

        except Exception as exc:

            last_error = exc

            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))

    raise RuntimeError(
        f"Bitget gagal: {type(last_error).__name__}: "
        f"{str(last_error)[:180]}"
    )


# ============================================================
# MULTI PROVIDER DATA
# ============================================================

@st.cache_data(
    ttl=CACHE_SECONDS,
    show_spinner=False,
)
def get_futures_history(inst_id):

    errors = []

    # ========================================================
    # PROVIDER 1 — OKX
    # ========================================================

    try:

        df = get_okx_candles(
            inst_id,
            limit=300,
        )

        return df, "OKX"

    except Exception as exc:

        errors.append(
            f"OKX: {str(exc)[:150]}"
        )

    # ========================================================
    # PROVIDER 2 — BITGET
    # ========================================================

    try:

        df = get_bitget_candles(
            inst_id,
            limit=200,
        )

        return df, "Bitget"

    except Exception as exc:

        errors.append(
            f"Bitget: {str(exc)[:150]}"
        )

    raise RuntimeError(
        " | ".join(errors)
    )


# ============================================================
# TECHNICAL FUNCTIONS
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

    if length <= 0 or len(values) < length:
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
                    max(0, i - length + 1):i + 1
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
# GPR PROFILE
#
# KITA KUNCI 1H SEBAGAI TARGET UTAMA
# ============================================================

def gpr_profile():

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
# CORE GPR
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
    band_mode="Pita GPR",
    band_sigma=1.96,
    atr_width=0.70,
    validation_candles=3,
):

    min_needed = max(
        training
        + vol_len
        + ema_slow
        + 10,
        80,
    )

    idx = (
        len(df)
        - 1
        - validation_candles
    )

    if idx < min_needed:

        raise ValueError(
            f"Candle hanya {len(df)}. "
            f"Minimal sekitar "
            f"{min_needed + validation_candles}."
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

    # ========================================================
    # LOG RETURN
    # ========================================================

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

    # ========================================================
    # MOMENTUM
    # ========================================================

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

    # ========================================================
    # VOLATILITY
    # ========================================================

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

    # ========================================================
    # RELATIVE VOLUME
    # ========================================================

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
        &
        (avg_volume > 0)
        &
        np.isfinite(avg_volume)
    )

    rel_volume[valid] = np.log(
        volume[valid]
        /
        avg_volume[valid]
    )

    # ========================================================
    # EMA
    # ========================================================

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

    # ========================================================
    # ATR
    # ========================================================

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
    # TRAINING DATA
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
            f"Training sample terlalu sedikit: {n}"
        )

    # ========================================================
    # GPR KERNEL MATRIX
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

    K += np.eye(n) * 1e-6

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

        K += np.eye(n) * 1e-4

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

            raise ValueError(
                "Matriks GPR tidak stabil."
            )

    # ========================================================
    # PROJECTION
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
                /
                max(
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

        try:

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

        except (
            np.linalg.LinAlgError,
            ValueError,
        ):

            predictive_variance = 1e-12

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

        if band_mode == "Kanal ATR":

            distance = (
                atr_width
                * float(atr[idx])
                * math.sqrt(
                    h + 1.0
                )
            )

            upper = (
                price_pred
                + distance
            )

            lower = max(
                1e-12,
                price_pred
                - distance,
            )

        else:

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
            idx + h + 1
        )

        actual_price = (

            float(
                close[real_idx]
            )

            if real_idx
            < len(df)

            else None
        )

        predictions.append({

            "step": h + 1,

            "is_future":
                actual_price is None,

            "price":
                round(
                    price_pred,
                    8,
                ),

            "actual_price":
                (
                    round(
                        actual_price,
                        8,
                    )
                    if actual_price
                    is not None
                    else None
                ),

            "upper":
                round(
                    upper,
                    8,
                ),

            "lower":
                round(
                    lower,
                    8,
                ),

            "sigma_cum":
                sigma_cum,
        })

        # ====================================================
        # Recursive GPR
        # ====================================================

        price_prev = price_pred

        ret_prev = pred_ret

        a_m = (
            2.0
            /
            (
                momentum_len
                + 1.0
            )
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
            (
                ema_fast
                + 1.0
            )
        )

        a_s = (
            2.0
            /
            (
                ema_slow
                + 1.0
            )
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
    # CURRENT STATE
    # ========================================================

    latest_actual_price = float(
        close[-1]
    )

    trend = (
        "BULLISH"
        if fast[-1] > slow[-1]
        else "BEARISH"
    )

    mom_now = float(
        momentum[-1]
    ) if np.isfinite(
        momentum[-1]
    ) else 0.0

    vol_now = float(
        volume[-1]
    )

    vol_avg = float(
        avg_volume[-1]
    ) if np.isfinite(
        avg_volume[-1]
    ) else 0.0

    rel_vol = (
        vol_now / vol_avg
        if vol_avg > 0
        else 0.0
    )

    final = predictions[-1]

    sigma_cum_final = math.sqrt(
        max(
            cumulative_variance,
            0.0,
        )
    )

    uncertainty = (
        100.0
        * sigma_cum_final
        /
        (
            abs(
                cumulative_return
            )
            + sigma_cum_final
            + 1e-10
        )
    )

    # ========================================================
    # VALIDATION T-3
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
        sum(
            validation_errors
        )
        /
        val_count
        if val_count > 0
        else 1.0
    )

    validation_hit_rate = (
        hits
        /
        val_count
        if val_count > 0
        else 0.0
    )

    return_from_now = (
        final["price"]
        /
        latest_actual_price
        - 1.0
    ) * 100.0

    return {

        "price":
            round(
                latest_actual_price,
                8,
            ),

        "projection_start_price":
            round(
                price_start,
                8,
            ),

        "trend":
            trend,

        "momentum":
            (
                "POSITIVE"
                if mom_now > 0
                else "NEGATIVE"
            ),

        "relative_volume":
            round(
                rel_vol,
                2,
            ),

        "gpr_return_pct":
            round(
                return_from_now,
                3,
            ),

        "uncertainty_pct":
            round(
                uncertainty,
                2,
            ),

        "cumulative_sigma":
            sigma_cum_final,

        "projected_price":
            final["price"],

        "horizon":
            horizon,

        "validation_mape":
            validation_mape,

        "validation_hit_rate":
            validation_hit_rate,

        "predictions":
            predictions,
    }


# ============================================================
# SCORING
#
# INI MASIH MEMPERTAHANKAN LOGIKA DASAR KAMU.
# NANTI BARU KITA OPTIMASI.
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

    mom_score = (
        10
        if result["momentum"]
        == "POSITIVE"
        else 0
    )

    ret_score = (
        25
        *
        max(
            0,
            min(
                result[
                    "gpr_return_pct"
                ]
                / 5.0,
                1.0,
            ),
        )
    )

    score = max(
        0,
        min(
            100,
            hit_score
            + mape_score
            + trend_score
            + mom_score
            + ret_score,
        ),
    )

    return round(
        score,
        1,
    )


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_one(
    inst_id,
    horizon,
):

    df, provider = (
        get_futures_history(
            inst_id
        )
    )

    params = gpr_profile()

    result = gpr_project(
        df,
        horizon=horizon,
        validation_candles=3,
        **params,
    )

    result["code"] = display_code(
        inst_id
    )

    result["instrument"] = inst_id

    result["market"] = (
        "Crypto Futures"
    )

    result["provider"] = provider

    result["score"] = calculate_score(
        result
    )

    # ========================================================
    # SIGNAL
    # ========================================================

    if (
        result["trend"]
        == "BULLISH"
        and result[
            "gpr_return_pct"
        ] > 0
        and result[
            "validation_hit_rate"
        ] >= 0.66
    ):

        result["signal"] = (
            "KANDIDAT"
        )

    elif (
        result[
            "gpr_return_pct"
        ] > 0
    ):

        result["signal"] = (
            "PANTAU"
        )

    else:

        result["signal"] = (
            "HINDARI"
        )

    return result


# ============================================================
# STREAMLIT CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMetric"] {
        background: #111c2e;
        border: 1px solid #26354a;
        padding: 12px 14px;
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #26354a;
        border-radius: 12px;
        overflow: hidden;
    }

    .note {
        font-size: 0.88rem;
        color: #9ca3af;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 GPR CRYPTO FUTURES SCANNER"
)

st.caption(
    "Target utama: USDT Perpetual · "
    "Timeframe 1H · GPR sebagai mesin ranking utama"
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

    quantity = st.selectbox(
        "Jumlah coin",
        [20, 30, 50, 70, 100, "Semua"],
        index=2,
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


# ============================================================
# CLEAR CACHE
# ============================================================

if clear_button:

    st.cache_data.clear()

    st.success(
        "Cache berhasil dibersihkan."
    )

    st.rerun()


# ============================================================
# UNIVERSE
# ============================================================

if quantity == "Semua":

    selected_symbols = (
        FUTURES_SYMBOLS.copy()
    )

else:

    selected_symbols = (
        FUTURES_SYMBOLS[
            :int(quantity)
        ]
    )


# ============================================================
# INFO
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Universe",
    len(selected_symbols),
)

c2.metric(
    "Timeframe",
    "1H",
)

c3.metric(
    "Horizon",
    f"{horizon}H",
)

c4.metric(
    "Provider",
    "OKX → Bitget",
)


st.info(
    "Mode data: public market data, "
    "tanpa API key. Universe futures dibuat "
    "statis agar scanner tidak bergantung pada "
    "endpoint instruments/exchangeInfo yang sebelumnya "
    "menghasilkan HTTP 403/451."
)


# ============================================================
# SESSION STATE
# ============================================================

if "gpr_results" not in st.session_state:
    st.session_state.gpr_results = []

if "gpr_errors" not in st.session_state:
    st.session_state.gpr_errors = []

if "gpr_last_scan" not in st.session_state:
    st.session_state.gpr_last_scan = None


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

    total = len(
        selected_symbols
    )

    completed = 0

    start_time = time.time()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                analyze_one,
                symbol,
                int(horizon),
            ): symbol

            for symbol
            in selected_symbols
        }

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

                    "code":
                        display_code(
                            symbol
                        ),

                    "instrument":
                        symbol,

                    "error":
                        str(exc)[:300],
                })

            completed += 1

            progress.progress(
                completed / total,
                text=(
                    f"Scanning "
                    f"{completed}/{total} · "
                    f"{display_code(symbol)}"
                ),
            )

            status.info(
                f"Berhasil: {len(results)} · "
                f"Gagal: {len(errors)}"
            )

    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda x: (
            float(
                x.get(
                    "score",
                    0
                )
            ),
            float(
                x.get(
                    "gpr_return_pct",
                    0
                )
            ),
        ),
        reverse=True,
    )

    st.session_state.gpr_results = (
        results
    )

    st.session_state.gpr_errors = (
        errors
    )

    st.session_state.gpr_last_scan = (
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    progress.progress(
        1.0,
        text="Scan selesai."
    )

    if results:

        st.success(
            f"Scan selesai: "
            f"{len(results)} berhasil, "
            f"{len(errors)} gagal "
            f"dalam {elapsed:.1f} detik."
        )

    else:

        st.error(
            "Tidak ada coin yang berhasil "
            "dianalisis."
        )


# ============================================================
# RESULTS
# ============================================================

results = (
    st.session_state.gpr_results
)

errors = (
    st.session_state.gpr_errors
)


# ============================================================
# TOP METRICS
# ============================================================

if results:

    qualified = [
        x
        for x in results
        if float(
            x.get(
                "score",
                0
            )
        )
        >= min_score
    ]

    candidates = [
        x
        for x in qualified
        if x.get(
            "signal"
        )
        == "KANDIDAT"
    ]

    bullish = [
        x
        for x in results
        if x.get(
            "trend"
        )
        == "BULLISH"
    ]

    best_return = max(
        (
            float(
                x[
                    "gpr_return_pct"
                ]
            )
            for x in results
        ),
        default=None,
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    m1.metric(
        "Berhasil dianalisis",
        len(results),
    )

    m2.metric(
        f"Lolos score ≥ {min_score}",
        len(qualified),
    )

    m3.metric(
        "KANDIDAT",
        len(candidates),
    )

    m4.metric(
        "GPR Return tertinggi",
        (
            f"{best_return:+.2f}%"
            if best_return
            is not None
            else "—"
        ),
    )


# ============================================================
# TABS
# ============================================================

rank_tab, detail_tab, error_tab = (
    st.tabs([
        "🏆 RANKING",
        "📈 DETAIL",
        "🧾 ERROR / DIAGNOSTIC",
    ])
)


# ============================================================
# RANKING
# ============================================================

with rank_tab:

    if not results:

        st.warning(
            "Belum ada hasil. "
            "Tekan SCAN FUTURES 1H."
        )

    else:

        query = st.text_input(
            "Cari coin",
            placeholder=(
                "Contoh: BTC, ETH, SOL"
            ),
        )

        view = [
            x
            for x in results
            if (
                not query.strip()
                or query.strip().upper()
                in x["code"].upper()
            )
        ]

        view = [
            x
            for x in view
            if float(
                x["score"]
            )
            >= min_score
        ]

        table = []

        for rank, item in enumerate(
            view,
            1,
        ):

            table.append({

                "Rank":
                    rank,

                "Coin":
                    item["code"],

                "Score":
                    item["score"],

                "Signal":
                    item["signal"],

                "GPR Return":
                    item[
                        "gpr_return_pct"
                    ],

                "Trend":
                    item["trend"],

                "Momentum":
                    item[
                        "momentum"
                    ],

                "Hit Rate":
                    f"{item['validation_hit_rate'] * 100:.0f}%",

                "MAPE":
                    f"{item['validation_mape'] * 100:.2f}%",

                "Uncertainty":
                    f"{item['uncertainty_pct']:.1f}%",

                "Harga":
                    item["price"],

                "Provider":
                    item["provider"],
            })

        if table:

            table_df = pd.DataFrame(
                table
            )

            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download Ranking CSV",
                table_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "gpr_crypto_futures_1h.csv"
                ),
                mime="text/csv",
            )

        else:

            st.info(
                "Tidak ada coin yang "
                "melewati minimum score."
            )


# ============================================================
# DETAIL
# ============================================================

with detail_tab:

    if not results:

        st.info(
            "Jalankan scan terlebih dahulu."
        )

    else:

        selected = st.selectbox(
            "Pilih coin",
            [
                x["code"]
                for x in results
            ],
        )

        item = next(
            x
            for x in results
            if x["code"]
            == selected
        )

        st.markdown(
            f"## {item['code']} "
            f"— {item['signal']}"
        )

        a, b, c, d = (
            st.columns(4)
        )

        a.metric(
            "Harga",
            f"{item['price']:,.8f}",
        )

        b.metric(
            "GPR Return",
            f"{item['gpr_return_pct']:+.2f}%",
        )

        c.metric(
            "Validation T-3",
            f"{item['validation_hit_rate'] * 100:.0f}%",
        )

        d.metric(
            "Uncertainty",
            f"{item['uncertainty_pct']:.1f}%",
        )

        st.write(
            f"**Provider:** "
            f"{item['provider']}"
        )

        st.write(
            f"**Trend:** "
            f"{item['trend']} · "
            f"**Momentum:** "
            f"{item['momentum']}"
        )

        st.write(
            f"**Score:** "
            f"{item['score']}/100"
        )

        preds = item.get(
            "predictions",
            [],
        )

        if preds:

            st.markdown(
                "### GPR Validation & Future Projection"
            )

            chart_data = []

            valid_count = sum(
                1
                for p in preds
                if not p[
                    "is_future"
                ]
            )

            chart_data.append({

                "Candle":
                    "Start T-3",

                "GPR":
                    item[
                        "projection_start_price"
                    ],

                "Actual":
                    item[
                        "projection_start_price"
                    ],

                "Upper":
                    item[
                        "projection_start_price"
                    ],

                "Lower":
                    item[
                        "projection_start_price"
                    ],
            })

            for p in preds:

                if p[
                    "is_future"
                ]:

                    future_no = (
                        p["step"]
                        - valid_count
                    )

                    label = (
                        f"Future +{future_no}H"
                    )

                else:

                    distance = (
                        valid_count
                        - p["step"]
                    )

                    if distance > 0:

                        label = (
                            f"Valid T-{distance}H"
                        )

                    else:

                        label = (
                            "T0"
                        )

                chart_data.append({

                    "Candle":
                        label,

                    "GPR":
                        p["price"],

                    "Actual":
                        p[
                            "actual_price"
                        ],

                    "Upper":
                        p["upper"],

                    "Lower":
                        p["lower"],
                })

            chart_df = (
                pd.DataFrame(
                    chart_data
                )
                .set_index(
                    "Candle"
                )
            )

            st.line_chart(
                chart_df
            )

            st.markdown(
                "### Projection Table"
            )

            detail_df = pd.DataFrame(
                preds
            )

            detail_df = (
                detail_df.drop(
                    columns=[
                        "is_future",
                        "sigma_cum",
                    ],
                    errors="ignore",
                )
            )

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# ERROR / DIAGNOSTIC
# ============================================================

with error_tab:

    st.markdown(
        "### Diagnostic"
    )

    st.write(
        f"**Last scan:** "
        f"{st.session_state.gpr_last_scan or '—'}"
    )

    st.write(
        f"**Universe:** "
        f"{len(selected_symbols)} coin"
    )

    st.write(
        f"**Berhasil:** "
        f"{len(results)}"
    )

    st.write(
        f"**Gagal:** "
        f"{len(errors)}"
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

    else:

        st.success(
            "Tidak ada error."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GPR Crypto Futures Scanner · "
    "Target utama 1H · "
    "Public market data · "
    "Eksperimental, bukan saran investasi."
)
