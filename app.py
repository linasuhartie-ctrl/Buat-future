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
    page_title="GPR Crypto Futures 1H",
    page_icon="📈",
    layout="wide",
)

CACHE_TTL = 300

# IMPORTANT:
# Untuk Indonesia gunakan endpoint resmi Bybit Indonesia.
# Tidak perlu API KEY untuk market data publik.
BYBIT_BASE = "https://api.bybit.id"

HEADERS = {
    "User-Agent": "Mozilla/5.0 GPR-Crypto-Futures/2.0",
    "Accept": "application/json",
}

# Kita sengaja TIDAK mengambil instruments-info setiap scan.
# Universe dibuat dari simbol USDT perpetual yang umum/liquid.
CORE_SYMBOLS = """
BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT DOGEUSDT ADAUSDT TRXUSDT
AVAXUSDT LINKUSDT SUIUSDT TONUSDT DOTUSDT BCHUSDT LTCUSDT SHIBUSDT
NEARUSDT APTUSDT ICPUSDT ETCUSDT XLMUSDT HBARUSDT FILUSDT ATOMUSDT
ARBUSDT OPUSDT INJUSDT AAVEUSDT UNIUSDT RENDERUSDT TAOUSDT IMXUSDT
GRTUSDT ALGOUSDT VETUSDT EOSUSDT XTZUSDT PEPEUSDT MKRUSDT
SEIUSDT TIAUSDT JUPUSDT WIFUSDT BONKUSDT FLOKIUSDT FETUSDT
ENAUSDT ONDOUSDT STXUSDT KASUSDT THETAUSDT CRVUSDT
LDOUSDT RUNEUSDT DYDXUSDT GMXUSDT COMPUSDT SNXUSDT
MANAUSDT SANDUSDT AXSUSDT GALAUSDT APEUSDT
WLDUSDT STRKUSDT ZKUSDT NOTUSDT BOMEUSDT
ORDIUSDT 1000SATSUSDT RATSUSDT
"""

SYMBOLS = list(dict.fromkeys(CORE_SYMBOLS.split()))


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)


def bybit_get(path, params, timeout=15):
    """
    Public GET request dengan retry ringan.
    Tidak menggunakan API key.
    """

    url = BYBIT_BASE + path

    last_error = None

    for attempt in range(3):
        try:
            r = session.get(
                url,
                params=params,
                timeout=timeout,
            )

            # rate limit
            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue

            # server/network issue
            r.raise_for_status()

            data = r.json()

            if data.get("retCode") != 0:
                raise RuntimeError(
                    f"Bybit retCode={data.get('retCode')} "
                    f"msg={data.get('retMsg')}"
                )

            return data

        except Exception as exc:
            last_error = exc
            time.sleep(1.0 + attempt)

    raise RuntimeError(str(last_error))


# ============================================================
# GET 1H FUTURES DATA
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_klines(symbol, limit=250):

    payload = bybit_get(
        "/v5/market/kline",
        {
            "category": "linear",
            "symbol": symbol,
            "interval": "60",
            "limit": limit,
        },
    )

    rows = payload.get("result", {}).get("list", [])

    if not rows:
        raise ValueError("Kline kosong")

    # Bybit mengembalikan newest -> oldest
    rows = list(reversed(rows))

    data = []

    for row in rows:
        data.append(
            {
                "time": pd.to_datetime(
                    int(row[0]),
                    unit="ms",
                    utc=True,
                ),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "turnover": float(row[6]),
            }
        )

    df = pd.DataFrame(data)

    return df.dropna().reset_index(drop=True)


# ============================================================
# GPR CORE
# ============================================================

def ema(series, length):
    return series.ewm(
        span=length,
        adjust=False,
        min_periods=1,
    ).mean()


def matern32_kernel(x, z, amplitude, scale):

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
            3.0 * max(float(distance2), 0.0)
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

    if length <= 0:
        return out

    if len(values) < length:
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
                    + (1 - alpha) * prev
                )

    return out


# ============================================================
# GPR PROFILE
# ============================================================

def gpr_profile():

    # FIXED:
    # Target utama hanya 1H.
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
# GPR PROJECTOR
# ============================================================

def gpr_project(
    df,
    horizon=10,
    validation_candles=3,
    training=35,
    momentum_len=5,
    vol_len=20,
    ema_fast=8,
    ema_slow=21,
    kernel_scale=1.6,
    noise_ratio=0.65,
    volume_fade=0.50,
):

    minimum = max(
        training
        + vol_len
        + ema_slow
        + 10,
        60,
    )

    idx = (
        len(df)
        - 1
        - validation_candles
    )

    if idx < minimum:
        raise ValueError(
            f"Candle tidak cukup: "
            f"{len(df)} / perlu {minimum + validation_candles}"
        )

    close = df["close"].astype(float).to_numpy()
    volume = df["volume"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    logret = np.full(
        len(close),
        np.nan,
    )

    logret[1:] = np.log(
        np.maximum(close[1:], 1e-12)
        /
        np.maximum(close[:-1], 1e-12)
    )

    ret_series = pd.Series(logret)

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

    # --------------------------------------------------------
    # EMA DIRECTION
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
        or
        not np.isfinite(
            atr[idx]
        )
        or
        not np.isfinite(
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
            or
            not np.isfinite(
                logret[target_idx]
            )
        ):
            continue

        X.append(
            [
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
            ]
        )

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
    # KERNEL MATRIX
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
    # RECURSIVE FORECAST
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
            +
            (
                vol_prev
                - vol_ref
            )
            * fade
        )

        x_star = np.array(
            [
                ret_prev / vol_ref,

                mom_prev / vol_ref,

                volume_prev * fade,

                assumed_vol / vol_ref
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
                /
                vol_ref,
            ]
        )

        kstar = np.array(
            [
                matern32_kernel(
                    X[i],
                    x_star,
                    signal_var,
                    kernel_scale,
                )
                for i in range(n)
            ]
        )

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

        # GPR confidence band
        upper = (
            price_pred
            * math.exp(
                1.96
                * sigma_cum
            )
        )

        lower = (
            price_pred
            * math.exp(
                -1.96
                * sigma_cum
            )
        )

        real_idx = (
            idx
            + h
            + 1
        )

        actual_price = (
            float(close[real_idx])
            if real_idx < len(df)
            else None
        )

        predictions.append(
            {
                "step": h + 1,
                "is_future": (
                    actual_price is None
                ),
                "price": price_pred,
                "actual_price": actual_price,
                "upper": upper,
                "lower": lower,
                "sigma_cum": sigma_cum,
            }
        )

        # recursive state update

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
            (
                1.0 - a_m
            )
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
            (
                1.0 - a_f
            )
            * fast_prev
        )

        slow_prev = (
            a_s * price_pred
            +
            (
                1.0 - a_s
            )
            * slow_prev
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

    vol_avg = (
        float(avg_volume[-1])
        if np.isfinite(
            avg_volume[-1]
        )
        else 0.0
    )

    relative_volume = (
        vol_now / vol_avg
        if vol_avg > 0
        else 0.0
    )

    # ========================================================
    # VALIDATION T-3
    # ========================================================

    errors = []
    hits = 0

    for p in predictions:

        if (
            not p["is_future"]
            and
            p["actual_price"]
            is not None
        ):

            error = (
                abs(
                    p["price"]
                    -
                    p["actual_price"]
                )
                /
                max(
                    p["actual_price"],
                    1e-12,
                )
            )

            errors.append(error)

            if (
                p["lower"]
                <=
                p["actual_price"]
                <=
                p["upper"]
            ):
                hits += 1

    val_count = len(errors)

    validation_mape = (
        np.mean(errors)
        if val_count
        else 1.0
    )

    validation_hit_rate = (
        hits / val_count
        if val_count
        else 0.0
    )

    final = predictions[-1]

    gpr_return = (
        final["price"]
        /
        latest_price
        - 1.0
    ) * 100

    sigma_final = math.sqrt(
        max(
            cumulative_variance,
            0.0,
        )
    )

    uncertainty = (
        100
        *
        sigma_final
        /
        (
            abs(
                cumulative_return
            )
            +
            sigma_final
            +
            1e-10
        )
    )

    return {
        "price": latest_price,
        "trend": trend,
        "momentum": (
            "POSITIVE"
            if mom_now > 0
            else "NEGATIVE"
        ),
        "relative_volume": relative_volume,
        "gpr_return_pct": gpr_return,
        "uncertainty_pct": uncertainty,
        "validation_mape": validation_mape,
        "validation_hit_rate": validation_hit_rate,
        "projected_price": final["price"],
        "predictions": predictions,
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
        +
        mape_score
        +
        trend_score
        +
        momentum_score
        +
        return_score
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


# ============================================================
# SINGLE SYMBOL
# ============================================================

def analyze_symbol(symbol):

    try:

        df = get_klines(
            symbol,
            limit=250,
        )

        if len(df) < 80:
            raise ValueError(
                f"Candle hanya {len(df)}"
            )

        result = gpr_project(
            df,
            horizon=10,
            validation_candles=3,
            **gpr_profile(),
        )

        result["symbol"] = symbol
        result["score"] = calculate_score(
            result
        )

        # ----------------------------------------------------
        # FUTURES SIGNAL
        # ----------------------------------------------------

        ret = result[
            "gpr_return_pct"
        ]

        trend = result[
            "trend"
        ]

        momentum = result[
            "momentum"
        ]

        hit = result[
            "validation_hit_rate"
        ]

        uncertainty = result[
            "uncertainty_pct"
        ]

        # LONG
        if (
            ret > 0
            and trend == "BULLISH"
            and momentum == "POSITIVE"
            and hit >= 0.66
        ):
            signal = "LONG"

        # SHORT
        elif (
            ret < 0
            and trend == "BEARISH"
            and momentum == "NEGATIVE"
            and hit >= 0.66
        ):
            signal = "SHORT"

        # WATCH
        elif abs(ret) >= 1.0:
            signal = "WATCH"

        else:
            signal = "AVOID"

        # confidence label
        if (
            result["score"] >= 80
            and uncertainty < 45
        ):
            quality = "A"

        elif (
            result["score"] >= 70
        ):
            quality = "B"

        else:
            quality = "C"

        result["signal"] = signal
        result["quality"] = quality

        return result

    except Exception as exc:

        return {
            "symbol": symbol,
            "error": str(exc)[:180],
        }


# ============================================================
# SCAN
# ============================================================

def scan_symbols(
    symbols,
    workers=3,
):

    results = []
    errors = []

    progress = st.progress(
        0,
        text="Memulai scanner..."
    )

    status = st.empty()

    # SENGAJA workers kecil.
    # Jangan 8-20 thread karena exchange gampang rate-limit.
    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                analyze_symbol,
                symbol,
            ): symbol
            for symbol in symbols
        }

        total = len(futures)

        for i, future in enumerate(
            as_completed(futures),
            1,
        ):

            result = future.result()

            if "error" in result:
                errors.append(result)

            else:
                results.append(result)

            progress.progress(
                i / total,
                text=(
                    f"Scanning "
                    f"{i}/{total} "
                    f"· berhasil "
                    f"{len(results)} "
                    f"· gagal "
                    f"{len(errors)}"
                ),
            )

            status.info(
                f"Last: {result['symbol']}"
            )

    progress.empty()
    status.empty()

    results.sort(
        key=lambda x: (
            x["score"],
            abs(x["gpr_return_pct"]),
        ),
        reverse=True,
    )

    return results, errors


# ============================================================
# UI
# ============================================================

st.title(
    "📈 GPR CRYPTO FUTURES SCANNER"
)

st.caption(
    "USDT Perpetual · 1H · "
    "GPR sebagai mesin ranking utama"
)

st.info(
    "Provider: Bybit Indonesia public market data. "
    "Scanner tidak membutuhkan API key."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header(
        "⚙️ Scanner"
    )

    scan_size = st.selectbox(
        "Jumlah coin",
        [20, 40, 60, 80, 100],
        index=1,
    )

    workers = st.slider(
        "Concurrent request",
        1,
        5,
        3,
        help=(
            "Jangan terlalu besar agar tidak "
            "terkena rate limit."
        ),
    )

    min_score = st.slider(
        "Minimum score",
        50,
        95,
        70,
        5,
    )

    run_scan = st.button(
        "🚀 SCAN FUTURES 1H",
        type="primary",
        use_container_width=True,
    )

    clear_cache = st.button(
        "🗑️ Clear cache",
        use_container_width=True,
    )

if clear_cache:

    st.cache_data.clear()

    st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "errors" not in st.session_state:
    st.session_state.errors = []


# ============================================================
# RUN
# ============================================================

if run_scan:

    selected_symbols = SYMBOLS[
        :scan_size
    ]

    results, errors = scan_symbols(
        selected_symbols,
        workers=workers,
    )

    st.session_state.results = results
    st.session_state.errors = errors


# ============================================================
# RESULTS
# ============================================================

results = st.session_state.results
errors = st.session_state.errors

if not results:

    st.warning(
        "Belum ada hasil. "
        "Tekan SCAN FUTURES 1H."
    )

    st.stop()


# ------------------------------------------------------------
# FILTER
# ------------------------------------------------------------

qualified = [
    x
    for x in results
    if x["score"] >= min_score
]

longs = [
    x
    for x in qualified
    if x["signal"] == "LONG"
]

shorts = [
    x
    for x in qualified
    if x["signal"] == "SHORT"
]

watch = [
    x
    for x in qualified
    if x["signal"] == "WATCH"
]


# ============================================================
# METRICS
# ============================================================

a, b, c, d = st.columns(4)

a.metric(
    "Scanned",
    len(results),
)

b.metric(
    "LONG",
    len(longs),
)

c.metric(
    "SHORT",
    len(shorts),
)

d.metric(
    "WATCH",
    len(watch),
)


# ============================================================
# TABLE
# ============================================================

st.subheader(
    "🏆 GPR Futures Ranking"
)

table = []

for rank, x in enumerate(
    qualified,
    1,
):

    table.append(
        {
            "Rank": rank,
            "Symbol": x["symbol"],
            "Score": x["score"],
            "Signal": x["signal"],
            "Quality": x["quality"],
            "GPR Return %": round(
                x["gpr_return_pct"],
                2,
            ),
            "T-3 Hit Rate": f"{x['validation_hit_rate'] * 100:.0f}%",
            "MAPE": f"{x['validation_mape'] * 100:.2f}%",
            "Trend": x["trend"],
            "Momentum": x["momentum"],
            "Rel Volume": round(
                x["relative_volume"],
                2,
            ),
            "Uncertainty": f"{x['uncertainty_pct']:.1f}%",
            "Price": x["price"],
            "GPR Target": x["projected_price"],
        }
    )

table_df = pd.DataFrame(
    table
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# LONG
# ============================================================

st.subheader(
    "🟢 LONG Candidates"
)

if longs:

    long_df = pd.DataFrame(
        [
            {
                "Symbol": x["symbol"],
                "Score": x["score"],
                "GPR Return %": round(
                    x["gpr_return_pct"],
                    2,
                ),
                "Hit Rate": f"{x['validation_hit_rate'] * 100:.0f}%",
                "Uncertainty": f"{x['uncertainty_pct']:.1f}%",
                "Price": x["price"],
                "GPR Target": x["projected_price"],
            }
            for x in longs
        ]
    )

    st.dataframe(
        long_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Belum ada LONG yang memenuhi filter."
    )


# ============================================================
# SHORT
# ============================================================

st.subheader(
    "🔴 SHORT Candidates"
)

if shorts:

    short_df = pd.DataFrame(
        [
            {
                "Symbol": x["symbol"],
                "Score": x["score"],
                "GPR Return %": round(
                    x["gpr_return_pct"],
                    2,
                ),
                "Hit Rate": f"{x['validation_hit_rate'] * 100:.0f}%",
                "Uncertainty": f"{x['uncertainty_pct']:.1f}%",
                "Price": x["price"],
                "GPR Target": x["projected_price"],
            }
            for x in shorts
        ]
    )

    st.dataframe(
        short_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Belum ada SHORT yang memenuhi filter."
    )


# ============================================================
# DETAIL
# ============================================================

st.subheader(
    "🔎 Detail GPR"
)

selected = st.selectbox(
    "Pilih coin",
    [x["symbol"] for x in results],
)

item = next(
    x
    for x in results
    if x["symbol"] == selected
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Price",
    f"{item['price']:,.6f}",
)

m2.metric(
    "GPR Return",
    f"{item['gpr_return_pct']:+.2f}%",
)

m3.metric(
    "Score",
    item["score"],
)

m4.metric(
    "Uncertainty",
    f"{item['uncertainty_pct']:.1f}%",
)


preds = item[
    "predictions"
]

chart = []

for p in preds:

    if p["is_future"]:

        label = (
            f"Future +{p['step']}"
        )

        actual = np.nan

    else:

        label = (
            f"Validation +{p['step']}"
        )

        actual = p[
            "actual_price"
        ]

    chart.append(
        {
            "Candle": label,
            "GPR": p["price"],
            "Actual": actual,
            "Upper": p["upper"],
            "Lower": p["lower"],
        }
    )

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
    f"⚠️ Error ({len(errors)})"
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
# DISCLAIMER
# ============================================================

st.caption(
    "GPR adalah model eksperimental. "
    "Score bukan probabilitas profit. "
    "Hasil bukan saran investasi."
)
