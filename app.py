import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="GPR Crypto Futures 1H Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CACHE_SECONDS = 180
LOOKBACK = 240
INTERVAL = "1h"

BINANCE_BASE = "https://fapi.binance.com"
BYBIT_BASE = "https://api.bybit.com"

# ============================================================
# GPR CORE — dipertahankan sebagai mesin utama screener
# ============================================================

def ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def matern32_kernel(x, z, amplitude, scale):
    d = x - z
    distance2 = (
        d[0] ** 2
        + 0.8 * d[1] ** 2
        + 0.6 * d[2] ** 2
        + 0.8 * d[3] ** 2
        + 0.6 * d[4] ** 2
    )
    u = math.sqrt(3.0 * max(float(distance2), 0.0)) / scale
    return amplitude * (1.0 + u) * math.exp(-u)


def pine_rma(values, length):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)

    if length <= 0 or len(values) < length:
        return out

    valid_idx = np.flatnonzero(np.isfinite(values))
    if len(valid_idx) < length:
        return out

    seed_indices = valid_idx[:length]
    seed_end = int(seed_indices[-1])
    out[seed_end] = float(np.mean(values[seed_indices]))

    alpha = 1.0 / length
    for i in range(seed_end + 1, len(values)):
        if np.isfinite(values[i]):
            prev = out[i - 1]
            if not np.isfinite(prev):
                prior = values[max(0, i - length + 1):i + 1]
                if np.isfinite(prior).sum() == length:
                    out[i] = float(np.mean(prior))
            else:
                out[i] = alpha * values[i] + (1.0 - alpha) * prev

    return out


def gpr_profile():
    # Khusus Crypto Futures 1H.
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


def gpr_project(
    df,
    horizon=5,
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
    min_needed = max(training + vol_len + ema_slow + 10, 60)
    idx = len(df) - 1 - validation_candles

    if idx < min_needed:
        raise ValueError(
            f"Data candle hanya {len(df)}; perlu minimal "
            f"{min_needed + validation_candles}."
        )

    close = df["close"].astype(float).to_numpy()
    volume = df["volume"].fillna(0).astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    logret = np.full(len(close), np.nan)
    logret[1:] = np.log(
        np.maximum(close[1:], 1e-12) /
        np.maximum(close[:-1], 1e-12)
    )

    ret_series = pd.Series(logret)
    momentum = (
        ret_series
        .ewm(span=momentum_len, adjust=False, min_periods=1)
        .mean()
        .to_numpy()
    )

    volatility = (
        ret_series
        .rolling(vol_len, min_periods=vol_len)
        .std(ddof=0)
        .to_numpy()
    )

    vol_base = (
        pd.Series(volatility)
        .rolling(training, min_periods=training)
        .mean()
        .to_numpy()
    )

    avg_volume = (
        pd.Series(volume)
        .rolling(vol_len, min_periods=vol_len)
        .mean()
        .to_numpy()
    )

    rel_volume = np.zeros(len(close))
    valid = (
        (volume > 0)
        & (avg_volume > 0)
        & np.isfinite(avg_volume)
    )
    rel_volume[valid] = np.log(
        volume[valid] / avg_volume[valid]
    )

    fast = ema(pd.Series(close), ema_fast).to_numpy()
    slow = ema(pd.Series(close), ema_slow).to_numpy()
    direction = (fast - slow) / np.maximum(close, 1e-12)

    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )
    tr[0] = high[0] - low[0]
    atr = pine_rma(tr, vol_len)

    vol_ref = max(
        float(vol_base[idx])
        if np.isfinite(vol_base[idx])
        else 0.0,
        0.0001,
    )

    if (
        not np.isfinite(volatility[idx])
        or not np.isfinite(atr[idx])
        or not np.isfinite(logret[idx])
    ):
        raise ValueError("Indikator belum siap pada candle terakhir.")

    X, y = [], []

    for r in range(training):
        target_idx = idx - training + 1 + r
        feature_idx = target_idx - 1

        if feature_idx < 1 or not np.isfinite(logret[target_idx]):
            continue

        X.append([
            logret[feature_idx] / vol_ref,
            momentum[feature_idx] / vol_ref,
            rel_volume[feature_idx],
            volatility[feature_idx] / vol_ref - 1.0,
            direction[feature_idx] / vol_ref,
        ])
        y.append(logret[target_idx])

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 10:
        raise ValueError("Sample training GPR terlalu sedikit.")

    signal_var = max(float(np.mean(y * y)), 1e-8)
    noise_var = signal_var * noise_ratio + 1e-10

    K = np.empty((n, n), dtype=float)

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
            K[i, j] = K[j, i] = value

    K += np.eye(n) * 1e-6

    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(
            L.T,
            np.linalg.solve(L, y),
        )
    except np.linalg.LinAlgError:
        try:
            K += np.eye(n) * 1e-4
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(
                L.T,
                np.linalg.solve(L, y),
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Matriks GPR tidak stabil secara numerik."
            ) from exc

    price_start = float(close[idx])
    price_prev = price_start
    ret_prev = float(logret[idx])
    mom_prev = float(momentum[idx])
    fast_prev = float(fast[idx])
    slow_prev = float(slow[idx])
    volume_prev = float(rel_volume[idx])
    vol_prev = float(volatility[idx])

    cumulative_return = 0.0
    cumulative_variance = 0.0
    predictions = []

    total_steps = horizon + validation_candles

    for h in range(total_steps):
        fade = volume_fade ** h

        assumed_vol = (
            vol_ref + (vol_prev - vol_ref) * fade
        )

        x_star = np.array([
            ret_prev / vol_ref,
            mom_prev / vol_ref,
            volume_prev * fade,
            assumed_vol / vol_ref - 1.0,
            (
                (fast_prev - slow_prev)
                / max(price_prev, 1e-12)
            ) / vol_ref,
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

        pred_ret = float(kstar @ alpha)

        try:
            solved = np.linalg.solve(L, kstar)
            predictive_variance = max(
                float(
                    signal_var
                    + noise_var
                    - solved @ solved
                ),
                1e-12,
            )
        except np.linalg.LinAlgError:
            predictive_variance = 1e-12

        cumulative_variance += predictive_variance
        cumulative_return += pred_ret

        price_pred = (
            price_start
            * math.exp(cumulative_return)
        )

        sigma_cum = math.sqrt(
            max(cumulative_variance, 0.0)
        )

        upper = price_pred * math.exp(
            band_sigma * sigma_cum
        )
        lower = price_pred * math.exp(
            -band_sigma * sigma_cum
        )

        real_idx = idx + h + 1
        actual_price = (
            float(close[real_idx])
            if real_idx < len(df)
            else None
        )

        predictions.append({
            "step": h + 1,
            "is_future": actual_price is None,
            "price": price_pred,
            "actual_price": actual_price,
            "upper": upper,
            "lower": lower,
            "sigma_cum": sigma_cum,
        })

        price_prev = price_pred
        ret_prev = pred_ret

        a_m = 2.0 / (momentum_len + 1.0)
        mom_prev = (
            a_m * pred_ret
            + (1.0 - a_m) * mom_prev
        )

        a_f = 2.0 / (ema_fast + 1.0)
        a_s = 2.0 / (ema_slow + 1.0)

        fast_prev = (
            a_f * price_pred
            + (1.0 - a_f) * fast_prev
        )
        slow_prev = (
            a_s * price_pred
            + (1.0 - a_s) * slow_prev
        )

    latest_actual_price = float(close[-1])

    trend = (
        "BULLISH"
        if fast[-1] > slow[-1]
        else "BEARISH"
    )

    mom_now = (
        float(momentum[-1])
        if np.isfinite(momentum[-1])
        else 0.0
    )

    vol_now = float(volume[-1])
    vol_avg = (
        float(avg_volume[-1])
        if np.isfinite(avg_volume[-1])
        else 0.0
    )

    rel_vol_now = (
        vol_now / vol_avg
        if vol_avg > 0
        else 0.0
    )

    final = predictions[-1]
    sigma_cum_final = math.sqrt(
        max(cumulative_variance, 0.0)
    )

    uncertainty = (
        100.0
        * sigma_cum_final
        / (
            abs(cumulative_return)
            + sigma_cum_final
            + 1e-10
        )
    )

    validation_errors = []
    hits = 0

    for p in predictions:
        if (
            not p["is_future"]
            and p["actual_price"] is not None
        ):
            err = (
                abs(p["price"] - p["actual_price"])
                / max(p["actual_price"], 1e-12)
            )
            validation_errors.append(err)

            if (
                p["lower"]
                <= p["actual_price"]
                <= p["upper"]
            ):
                hits += 1

    val_count = len(validation_errors)

    validation_mape = (
        sum(validation_errors) / val_count
        if val_count
        else 1.0
    )

    validation_hit_rate = (
        hits / val_count
        if val_count
        else 0.0
    )

    return_from_now = (
        final["price"]
        / latest_actual_price
        - 1.0
    ) * 100.0

    return {
        "price": latest_actual_price,
        "projection_start_price": price_start,
        "trend": trend,
        "momentum": (
            "POSITIVE"
            if mom_now > 0
            else "NEGATIVE"
        ),
        "relative_volume": rel_vol_now,
        "gpr_return_pct": return_from_now,
        "uncertainty_pct": uncertainty,
        "cumulative_sigma": sigma_cum_final,
        "projected_price": final["price"],
        "horizon": horizon,
        "validation_mape": validation_mape,
        "validation_hit_rate": validation_hit_rate,
        "predictions": predictions,
    }


# ============================================================
# FUTURES DATA PROVIDERS
# ============================================================

session = requests.Session()
session.headers.update({
    "User-Agent": "GPR-Crypto-Futures-Scanner/1.0"
})


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_binance_symbols():
    r = session.get(
        f"{BINANCE_BASE}/fapi/v1/exchangeInfo",
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    rows = []
    for s in data.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
        ):
            rows.append(s["symbol"])

    return sorted(set(rows))


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_bybit_symbols():
    symbols = []
    cursor = None

    while True:
        params = {
            "category": "linear",
            "status": "Trading",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor

        r = session.get(
            f"{BYBIT_BASE}/v5/market/instruments-info",
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        result = data.get("result", {})
        for s in result.get("list", []):
            if (
                s.get("status") == "Trading"
                and s.get("contractType") == "LinearPerpetual"
                and s.get("quoteCoin") == "USDT"
            ):
                symbols.append(s["symbol"])

        cursor = result.get("nextPageCursor")
        if not cursor:
            break

    return sorted(set(symbols))


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_binance_klines(symbol, limit=LOOKBACK):
    r = session.get(
        f"{BINANCE_BASE}/fapi/v1/klines",
        params={
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": limit,
        },
        timeout=15,
    )
    r.raise_for_status()
    rows = r.json()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume",
        "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])

    for col in [
        "open", "high", "low", "close",
        "volume", "quote_volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    return df[
        [
            "time", "open", "high", "low",
            "close", "volume", "quote_volume"
        ]
    ].dropna().reset_index(drop=True)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_bybit_klines(symbol, limit=LOOKBACK):
    r = session.get(
        f"{BYBIT_BASE}/v5/market/kline",
        params={
            "category": "linear",
            "symbol": symbol,
            "interval": "60",
            "limit": min(limit, 1000),
        },
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()

    if payload.get("retCode") != 0:
        raise RuntimeError(
            payload.get("retMsg", "Bybit API error")
        )

    rows = payload.get("result", {}).get("list", [])

    if not rows:
        return pd.DataFrame()

    # Bybit mengembalikan candle terbaru lebih dulu.
    rows = list(reversed(rows))

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low",
        "close", "volume", "turnover"
    ])

    for col in [
        "open", "high", "low",
        "close", "volume", "turnover"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["open_time"]),
        unit="ms",
        utc=True,
    )

    # Samakan volume quote agar filter likuiditas
    # bisa dibandingkan dengan Binance.
    df["quote_volume"] = df["turnover"]

    return df[
        [
            "time", "open", "high", "low",
            "close", "volume", "quote_volume"
        ]
    ].dropna().reset_index(drop=True)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_binance_24h():
    r = session.get(
        f"{BINANCE_BASE}/fapi/v1/ticker/24hr",
        timeout=15,
    )
    r.raise_for_status()

    rows = r.json()
    out = {}

    for x in rows:
        symbol = x.get("symbol")
        if symbol:
            out[symbol] = {
                "quote_volume_24h": float(
                    x.get("quoteVolume", 0)
                ),
                "change_24h": float(
                    x.get("priceChangePercent", 0)
                ),
            }

    return out


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def get_bybit_24h():
    r = session.get(
        f"{BYBIT_BASE}/v5/market/tickers",
        params={"category": "linear"},
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()

    if payload.get("retCode") != 0:
        raise RuntimeError(
            payload.get("retMsg", "Bybit ticker error")
        )

    out = {}

    for x in payload.get("result", {}).get("list", []):
        symbol = x.get("symbol")
        if symbol:
            out[symbol] = {
                "quote_volume_24h": float(
                    x.get("turnover24h", 0)
                ),
                "change_24h": float(
                    x.get("price24hPcnt", 0)
                ) * 100.0,
                "funding_rate": float(
                    x.get("fundingRate", 0)
                ) * 100.0,
            }

    return out


def get_symbols(exchange):
    if exchange == "Binance":
        return get_binance_symbols()
    return get_bybit_symbols()


def get_klines(exchange, symbol):
    if exchange == "Binance":
        return get_binance_klines(symbol)
    return get_bybit_klines(symbol)


def get_market_meta(exchange):
    if exchange == "Binance":
        return get_binance_24h()
    return get_bybit_24h()


# ============================================================
# ANALYSIS / SCORING
# ============================================================

def analyze_one(exchange, symbol, horizon, meta):
    df = get_klines(exchange, symbol)

    if df.empty or len(df) < 80:
        raise ValueError(
            f"Data 1H kurang: {len(df)} candle."
        )

    params = gpr_profile()

    result = gpr_project(
        df,
        horizon=horizon,
        validation_candles=3,
        **params,
    )

    m = meta.get(symbol, {})

    quote_volume_24h = float(
        m.get("quote_volume_24h", 0)
    )
    change_24h = float(
        m.get("change_24h", 0)
    )
    funding_rate = float(
        m.get("funding_rate", 0)
    )

    # --------------------------------------------------------
    # SCORE: GPR tetap menjadi inti.
    #
    # 30  = validation hit rate
    # 20  = validation error
    # 15  = trend
    # 10  = momentum
    # 25  = projected return
    # --------------------------------------------------------
    hit_score = (
        result["validation_hit_rate"] * 30
    )

    mape_score = max(
        0,
        20 - (
            result["validation_mape"]
            * 100
            * 4
        ),
    )

    trend_score = (
        15
        if result["trend"] == "BULLISH"
        else 0
    )

    mom_score = (
        10
        if result["momentum"] == "POSITIVE"
        else 0
    )

    ret_score = 25 * max(
        0,
        min(
            result["gpr_return_pct"] / 5.0,
            1.0,
        ),
    )

    gpr_score = max(
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

    # Likuiditas hanya sebagai FILTER, bukan pengganti GPR.
    liquidity_ok = quote_volume_24h >= 10_000_000

    if (
        result["trend"] == "BULLISH"
        and result["gpr_return_pct"] > 0
        and result["validation_hit_rate"] >= 0.66
        and liquidity_ok
    ):
        signal = "LONG CANDIDATE"
    elif (
        result["gpr_return_pct"] > 0
        and liquidity_ok
    ):
        signal = "WATCH"
    elif (
        result["trend"] == "BEARISH"
        and result["gpr_return_pct"] < 0
        and liquidity_ok
    ):
        signal = "SHORT CANDIDATE"
    else:
        signal = "AVOID"

    result.update({
        "symbol": symbol,
        "exchange": exchange,
        "gpr_score": round(gpr_score, 1),
        "signal": signal,
        "quote_volume_24h": quote_volume_24h,
        "change_24h": change_24h,
        "funding_rate": funding_rate,
        "liquidity_ok": liquidity_ok,
    })

    return result


# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}
#MainMenu, footer {visibility: hidden;}
[data-testid="stMetric"] {
    border: 1px solid #26354a;
    padding: 12px 14px;
    border-radius: 12px;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 GPR CRYPTO FUTURES SCANNER")
st.caption(
    "Target utama: USDT Perpetual · Timeframe 1H · "
    "GPR sebagai mesin ranking utama"
)

with st.sidebar:
    st.subheader("⚙️ Scanner")

    exchange = st.selectbox(
        "Exchange",
        ["Binance", "Bybit"],
    )

    horizon = st.selectbox(
        "Horizon GPR",
        [3, 5, 10, 20],
        index=1,
        help="Jumlah candle 1H yang diproyeksikan.",
    )

    min_score = st.slider(
        "Minimum GPR score",
        40,
        90,
        65,
        5,
    )

    min_volume = st.number_input(
        "Minimum volume 24H (USDT)",
        min_value=0.0,
        value=10_000_000.0,
        step=5_000_000.0,
    )

    max_symbols = st.selectbox(
        "Jumlah aset dipindai",
        [50, 100, 200, 300, 500, 1000],
        index=2,
    )

    workers = st.slider(
        "Concurrent workers",
        2,
        12,
        8,
    )

    scan_btn = st.button(
        "🚀 SCAN FUTURES 1H",
        type="primary",
        use_container_width=True,
    )

    clear_btn = st.button(
        "🗑️ Clear cache",
        use_container_width=True,
    )

if clear_btn:
    st.cache_data.clear()
    st.rerun()

# Load universe
try:
    symbols = get_symbols(exchange)
except Exception as exc:
    st.error(
        f"Gagal mengambil daftar futures {exchange}: "
        f"{type(exc).__name__}: {str(exc)[:180]}"
    )
    st.stop()

try:
    meta = get_market_meta(exchange)
except Exception:
    meta = {}

# Apply liquidity filter BEFORE GPR to avoid wasting API calls.
liquid_symbols = [
    s for s in symbols
    if float(
        meta.get(s, {}).get(
            "quote_volume_24h",
            0
        )
    ) >= min_volume
]

liquid_symbols = sorted(
    liquid_symbols,
    key=lambda s: meta.get(s, {}).get(
        "quote_volume_24h", 0
    ),
    reverse=True,
)

scan_symbols = liquid_symbols[:max_symbols]

c1, c2, c3, c4 = st.columns(4)
c1.metric("USDT Perpetual", len(symbols))
c2.metric("Lolos likuiditas", len(liquid_symbols))
c3.metric("Target scan", len(scan_symbols))
c4.metric("Timeframe", "1H")

st.info(
    f"{exchange}: {len(scan_symbols)} kontrak akan diproses. "
    "Universe difilter likuiditas dulu; setelah itu GPR menghitung "
    "ranking, validasi T-3, return proyeksi, trend, dan momentum."
)

if scan_btn:
    progress = st.progress(
        0,
        text="Memulai scan..."
    )
    status = st.empty()

    results = []
    errors = []

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = {
            executor.submit(
                analyze_one,
                exchange,
                symbol,
                int(horizon),
                meta,
            ): symbol
            for symbol in scan_symbols
        }

        done = 0

        for future in as_completed(futures):
            symbol = futures[future]

            try:
                results.append(
                    future.result()
                )
            except Exception as exc:
                errors.append({
                    "symbol": symbol,
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{str(exc)[:180]}"
                    ),
                })

            done += 1
            progress.progress(
                done / max(1, len(scan_symbols)),
                text=f"Scan {done}/{len(scan_symbols)} · {symbol}",
            )

            status.info(
                f"Berhasil: {len(results)} · "
                f"Gagal: {len(errors)}"
            )

    if results:
        results.sort(
            key=lambda x: (
                x["gpr_score"],
                x["gpr_return_pct"],
            ),
            reverse=True,
        )

    st.session_state["gpr_results"] = results
    st.session_state["gpr_errors"] = errors
    st.session_state["gpr_exchange"] = exchange
    st.session_state["gpr_horizon"] = horizon

    progress.progress(
        1.0,
        text="Scan selesai."
    )

results = st.session_state.get(
    "gpr_results",
    []
)

errors = st.session_state.get(
    "gpr_errors",
    []
)

if not results:
    st.markdown("### 🎯 Belum ada hasil")
    st.write(
        "Tekan **SCAN FUTURES 1H** untuk menjalankan screener."
    )
    st.stop()

qualified = [
    x for x in results
    if x["gpr_score"] >= min_score
]

longs = [
    x for x in qualified
    if x["signal"] == "LONG CANDIDATE"
]

shorts = [
    x for x in qualified
    if x["signal"] == "SHORT CANDIDATE"
]

best = results[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Aset dianalisis",
    len(results),
)
m2.metric(
    f"GPR ≥ {min_score}",
    len(qualified),
)
m3.metric(
    "Long Candidate",
    len(longs),
)
m4.metric(
    "Top GPR",
    f"{best['symbol']} · {best['gpr_score']:.1f}",
)

tab1, tab2, tab3 = st.tabs([
    "🏆 Ranking",
    "📊 Detail GPR",
    "🧾 Error",
])

with tab1:
    view = pd.DataFrame([
        {
            "Rank": i + 1,
            "Symbol": x["symbol"],
            "GPR Score": x["gpr_score"],
            "Signal": x["signal"],
            "GPR Return %": x["gpr_return_pct"],
            "T-3 Hit": f"{x['validation_hit_rate']*100:.0f}%",
            "MAPE": f"{x['validation_mape']*100:.2f}%",
            "Trend": x["trend"],
            "Momentum": x["momentum"],
            "Vol 24H USDT": x["quote_volume_24h"],
            "24H %": x["change_24h"],
            "Funding %": x["funding_rate"],
            "Price": x["price"],
            "Projected": x["projected_price"],
            "Uncertainty %": x["uncertainty_pct"],
        }
        for i, x in enumerate(qualified)
    ])

    if not view.empty:
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download ranking CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"gpr_{exchange.lower()}_futures_1h.csv"
            ),
            mime="text/csv",
        )
    else:
        st.warning(
            "Tidak ada aset yang memenuhi minimum GPR score."
        )

with tab2:
    symbols_ready = [
        x["symbol"]
        for x in results
    ]

    selected = st.selectbox(
        "Pilih aset",
        symbols_ready,
    )

    item = next(
        x for x in results
        if x["symbol"] == selected
    )

    a, b, c, d, e = st.columns(5)
    a.metric(
        "Harga",
        f"{item['price']:,.6g}",
    )
    b.metric(
        "GPR Score",
        f"{item['gpr_score']:.1f}",
    )
    c.metric(
        "GPR Return",
        f"{item['gpr_return_pct']:+.2f}%",
    )
    d.metric(
        "T-3 Hit",
        f"{item['validation_hit_rate']*100:.0f}%",
    )
    e.metric(
        "Uncertainty",
        f"{item['uncertainty_pct']:.1f}%",
    )

    st.write(
        f"**Signal:** {item['signal']} · "
        f"**Trend:** {item['trend']} · "
        f"**Momentum:** {item['momentum']} · "
        f"**24H volume:** "
        f"{item['quote_volume_24h']:,.0f} USDT"
    )

    preds = item["predictions"]

    chart_rows = []

    for p in preds:
        if p["is_future"]:
            label = f"Future +{p['step'] - 3}"
        else:
            label = f"T-{3 - p['step']}"

        chart_rows.append({
            "Candle": label,
            "GPR": p["price"],
            "Actual": p["actual_price"],
            "Upper": p["upper"],
            "Lower": p["lower"],
        })

    chart_df = pd.DataFrame(
        chart_rows
    ).set_index("Candle")

    st.line_chart(chart_df)

    detail_df = pd.DataFrame([
        {
            "Step": p["step"],
            "Future": p["is_future"],
            "GPR Price": p["price"],
            "Actual": p["actual_price"],
            "Upper": p["upper"],
            "Lower": p["lower"],
            "Sigma": p["sigma_cum"],
        }
        for p in preds
    ])

    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    if errors:
        st.dataframe(
            pd.DataFrame(errors),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success(
            "Tidak ada error pada aset yang diproses."
        )

st.caption(
    "GPR adalah model eksperimental untuk ranking/screening, "
    "bukan probabilitas profit dan bukan sinyal trading otomatis. "
    "Versi ini hanya memakai endpoint market-data publik; "
    "tidak membutuhkan API key dan tidak mengeksekusi order."
)
