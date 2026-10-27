import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
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
    font-weight: 700;
}

.signal-short {
    color: #e53935;
    font-weight: 700;
}

.metric-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    background: #fafafa;
    min-height: 120px;
}

.metric-title {
    font-size: 0.85rem;
    color: #777;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
}

.good-box {
    padding: 14px;
    border-radius: 10px;
    background: #eaf7ef;
    border: 1px solid #b9e4c8;
}

.warn-box {
    padding: 14px;
    border-radius: 10px;
    background: #fff8df;
    border: 1px solid #f0dc8a;
}

.error-box {
    padding: 14px;
    border-radius: 10px;
    background: #fdecec;
    border: 1px solid #efb1b1;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIG
# ============================================================

OKX_BASE = "https://www.okx.com"

# Static universe.
# Tidak menggunakan endpoint instruments agar tidak tergantung
# endpoint yang sebelumnya bermasalah.
UNIVERSE = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX",
    "LINK", "DOT", "TRX", "SUI", "TON", "SHIB", "LTC", "BCH",
    "NEAR", "APT", "ICP", "FIL", "ATOM", "ETC", "ARB", "OP",
    "INJ", "IMX", "AAVE", "MKR", "RUNE", "UNI", "LDO", "STX",
    "SEI", "TIA", "JUP", "WIF", "PEPE", "BONK", "FLOKI", "WLD",
    "ENA", "ONDO", "PYTH", "JTO", "STRK", "NOT", "ORDI", "SATS",
    "SAND", "MANA", "AXS", "GALA", "EOS", "XLM", "VET", "ALGO",
    "HBAR", "FTM", "APE", "CRV", "COMP", "SNX", "DYDX", "1INCH",
    "KAS", "TAO", "FET", "RENDER", "MANTA", "DYM", "WOO", "JASMY",
    "KAVA", "MINA", "CELO", "ZRX", "QTUM"
]



# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Scanner Settings")

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1H", "15m"],
    index=0
)

horizon = st.sidebar.selectbox(
    "Horizon GPR",
    [5, 10, 15, 20],
    index=1,
    format_func=lambda x: (
        f"{x} candle / {x} jam" if timeframe == "1H"
        else f"{x} candle / {x * 15 / 60:g} jam"
    )
)

coin_count = st.sidebar.selectbox(
    "Jumlah coin",
    [20, 30, 50, 79],
    index=3
)

minimum_score = st.sidebar.slider(
    "Minimum score",
    min_value=50,
    max_value=95,
    value=70,
    step=1
)

parallel_workers = st.sidebar.slider(
    "Parallel workers",
    min_value=1,
    max_value=8,
    value=5,
    step=1
)

st.sidebar.divider()

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
    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("📈 GPR CRYPTO FUTURES SCANNER")

st.caption(
    f"Target utama: USDT Perpetual · Timeframe {timeframe} · "
    f"GPR sebagai mesin ranking utama"
)


# ============================================================
# DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_okx_candles(symbol, bar="1H", limit=220):

    inst_id = f"{symbol}-USDT-SWAP"

    url = f"{OKX_BASE}/api/v5/market/candles"

    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": str(limit)
    }

    r = requests.get(
        url,
        params=params,
        timeout=12,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    r.raise_for_status()

    data = r.json()

    if data.get("code") != "0":
        raise RuntimeError(
            f"OKX code={data.get('code')} "
            f"msg={data.get('msg')}"
        )

    rows = data.get("data", [])

    if not rows:
        raise RuntimeError("No candle data")

    # OKX candle:
    # [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
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

    df["datetime"] = pd.to_datetime(
        df["ts"],
        unit="ms",
        utc=True
    )

    df = df.sort_values("datetime")

    # Hanya candle yang sudah confirmed
    if "confirm" in df.columns:
        df = df[df["confirm"].astype(str) == "1"]

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    if len(df) < 100:
        raise RuntimeError(
            f"Insufficient candles: {len(df)}"
        )

    return df.reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_features(df):

    x = df.copy()

    close = x["close"]
    volume = x["volume"]

    x["rsi14"] = calculate_rsi(
        close,
        14
    )

    x["mom_6"] = (
        close / close.shift(6) - 1
    ) * 100

    x["mom_12"] = (
        close / close.shift(12) - 1
    ) * 100

    x["mom_24"] = (
        close / close.shift(24) - 1
    ) * 100

    x["ema12"] = close.ewm(
        span=12,
        adjust=False
    ).mean()

    x["ema24"] = close.ewm(
        span=24,
        adjust=False
    ).mean()

    x["ema12_dist"] = (
        close / x["ema12"] - 1
    ) * 100

    x["ema24_dist"] = (
        close / x["ema24"] - 1
    ) * 100

    vol_mean = volume.rolling(30).mean()
    vol_std = volume.rolling(30).std()

    x["volume_z"] = (
        (volume - vol_mean) /
        vol_std.replace(0, np.nan)
    )

    # Volatility
    x["atr_pct"] = (
        (x["high"] - x["low"]) /
        close
    ) * 100

    # Trend
    x["ema12_24_spread"] = (
        x["ema12"] / x["ema24"] - 1
    ) * 100

    return x


# ============================================================
# GPR MODEL
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


def train_gpr(df, horizon_bars):

    data = add_features(df)

    # Future return
    data["future_return"] = (
        data["close"].shift(-horizon_bars) /
        data["close"] - 1
    ) * 100

    model_data = data.dropna(
        subset=FEATURES + ["future_return"]
    ).copy()

    if len(model_data) < 80:
        raise RuntimeError(
            f"Not enough training data: {len(model_data)}"
        )

    # Batasi training agar ringan
    model_data = model_data.tail(150)

    X = model_data[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = model_data["future_return"]

    valid = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 60:
        raise RuntimeError(
            "Insufficient valid training rows"
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0)
        * RBF(length_scale=1.0)
        + WhiteKernel(noise_level=0.5)
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

    latest = data.dropna(
        subset=FEATURES
    ).iloc[-1]

    X_latest = latest[FEATURES].values.reshape(
        1, -1
    )

    X_latest_scaled = scaler.transform(
        X_latest
    )

    pred, std = model.predict(
        X_latest_scaled,
        return_std=True
    )

    prediction = float(pred[0])
    uncertainty = float(std[0])

    return (
        prediction,
        uncertainty,
        latest,
        model
    )


# ============================================================
# SCORE ENGINE
# ============================================================

def calculate_score(
    prediction,
    uncertainty,
    latest
):

    # Magnitude component
    magnitude = min(
        abs(prediction) / 5.0,
        1.0
    )

    # Confidence component
    confidence = 1 / (
        1 + max(uncertainty, 0)
    )

    # RSI sanity
    rsi = float(latest["rsi14"])

    if prediction > 0:
        rsi_component = (
            1.0 if 45 <= rsi <= 70
            else 0.5
        )
    else:
        rsi_component = (
            1.0 if 30 <= rsi <= 55
            else 0.5
        )

    # Momentum alignment
    mom_values = [
        latest["mom_6"],
        latest["mom_12"],
        latest["mom_24"]
    ]

    avg_mom = np.nanmean(
        mom_values
    )

    if prediction > 0:
        momentum_alignment = (
            1.0 if avg_mom > 0
            else 0.4
        )
    else:
        momentum_alignment = (
            1.0 if avg_mom < 0
            else 0.4
        )

    score = (
        magnitude * 40
        + confidence * 25
        + rsi_component * 15
        + momentum_alignment * 20
    )

    score = float(
        np.clip(score, 0, 100)
    )

    return score


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_symbol(
    symbol,
    bar,
    horizon_bars
):

    df = get_okx_candles(
        symbol,
        bar=bar,
        limit=220
    )

    prediction, uncertainty, latest, model = train_gpr(
        df,
        horizon_bars
    )

    score = calculate_score(
        prediction,
        uncertainty,
        latest
    )

    signal = (
        "LONG"
        if prediction >= 0
        else "SHORT"
    )

    confidence = (
        abs(prediction) /
        max(uncertainty, 0.01)
    )

    result = {
        "COIN": symbol,
        "SIGNAL": signal,
        "GPR SCORE": score,
        "GPR RETURN": prediction,
        "UNCERTAINTY": uncertainty,
        "CONFIDENCE": confidence,
        "PRICE": float(latest["close"]),
        "RSI": float(latest["rsi14"]),
        "MOM 6H": float(latest["mom_6"]),
        "MOM 12H": float(latest["mom_12"]),
        "MOM 24H": float(latest["mom_24"]),
        "VOL Z": float(latest["volume_z"]),
        "EMA12 DIST": float(latest["ema12_dist"]),
        "EMA24 DIST": float(latest["ema24_dist"]),
        "CANDLE": latest["datetime"]
    }

    return result


# ============================================================
# SCAN
# ============================================================

def run_scan(
    symbols,
    timeframe,
    horizon_bars,
    workers
):

    bar = "1H" if timeframe == "1H" else "15m"

    results = []
    errors = []

    progress = st.progress(0)

    status = st.empty()

    total = len(symbols)

    # --------------------------------------------------------
    # Sequential execution.
    # Lebih stabil pada Streamlit Cloud daripada terlalu
    # banyak request paralel.
    # --------------------------------------------------------

    for i, symbol in enumerate(symbols):

        status.write(
            f"Scanning {symbol} "
            f"({i + 1}/{total})..."
        )

        try:

            result = analyze_symbol(
                symbol,
                bar,
                horizon_bars
            )

            results.append(result)

        except Exception as e:

            errors.append({
                "Symbol": symbol,
                "Status": str(e)
            })

        progress.progress(
            int((i + 1) / total * 100)
        )

        # Sedikit jeda agar tidak terlalu agresif
        time.sleep(0.03)

    progress.empty()
    status.empty()

    return results, errors


# ============================================================
# DISPLAY METRIC
# ============================================================

def metric_card(title, value, subtitle=""):

    with st.container(border=True):

        st.markdown(
            f'<div class="metric-title">{title}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="metric-value">{value}</div>',
            unsafe_allow_html=True
        )

        if subtitle:
            st.caption(subtitle)


# ============================================================
# RUN SCAN
# ============================================================

if scan_button:

    selected_symbols = UNIVERSE[:coin_count]

    st.info(
        f"Mode data: public market data, tanpa API key. "
        f"Provider OKX · {timeframe} · "
        f"Horizon {horizon} candle."
    )

    start_time = time.time()

    with st.spinner(
        f"Menjalankan GPR Futures Scanner {timeframe}..."
    ):

        results, errors = run_scan(
            selected_symbols,
            timeframe,
            horizon,
            parallel_workers
        )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    if results:

        df_result = pd.DataFrame(results)

        df_result = df_result.sort_values(
            "GPR SCORE",
            ascending=False
        ).reset_index(drop=True)

        df_result["Rank"] = (
            np.arange(len(df_result)) + 1
        )

        # ----------------------------------------------------
        # SAVE SESSION
        # ----------------------------------------------------

        st.session_state["results"] = df_result
        st.session_state["errors"] = pd.DataFrame(errors)
        st.session_state["timeframe"] = timeframe
        st.session_state["horizon"] = horizon
        st.session_state["elapsed"] = elapsed

    else:

        st.error(
            "Tidak ada coin yang berhasil dianalisis."
        )

        if errors:
            st.dataframe(
                pd.DataFrame(errors),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# SHOW RESULTS
# ============================================================

if "results" in st.session_state:

    df = st.session_state["results"]
    errors_df = st.session_state.get(
        "errors",
        pd.DataFrame()
    )

    active_tf = st.session_state["timeframe"]
    active_horizon = st.session_state["horizon"]
    elapsed = st.session_state["elapsed"]

    candidates = df[
        df["GPR SCORE"] >= minimum_score
    ].copy()

    # ========================================================
    # SUMMARY
    # ========================================================

    st.success(
        f"Scan selesai: {len(df)} berhasil, "
        f"{len(errors_df)} gagal dalam {elapsed:.1f} detik."
    )

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
            if active_tf == "1H"
            else active_horizon * 0.25
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

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "BERHASIL DIANALISIS",
            len(df)
        )

    with c2:
        metric_card(
            f"LOLOS SCORE ≥ {minimum_score}",
            len(candidates)
        )

    with c3:
        metric_card(
            "KANDIDAT",
            min(len(candidates), 10)
        )

    with c4:

        if len(df) > 0:

            top_return = df.iloc[0]["GPR RETURN"]

            metric_card(
                "GPR RETURN TERTINGGI",
                f"{top_return:+.2f}%",
                f"Score {df.iloc[0]['GPR SCORE']:.1f}"
            )

    # ========================================================
    # RANKING
    # ========================================================

    st.divider()

    st.subheader("🏆 GPR Futures Ranking")

    display_df = df[
        [
            "Rank",
            "COIN",
            "SIGNAL",
            "GPR SCORE",
            "GPR RETURN",
            "UNCERTAINTY",
            "CONFIDENCE",
            "PRICE",
            "RSI",
            "MOM 6H",
            "MOM 12H",
            "MOM 24H",
            "VOL Z"
        ]
    ].copy()

    display_df["GPR SCORE"] = display_df[
        "GPR SCORE"
    ].round(1)

    display_df["GPR RETURN"] = display_df[
        "GPR RETURN"
    ].map(lambda x: f"{x:+.2f}%")

    display_df["UNCERTAINTY"] = display_df[
        "UNCERTAINTY"
    ].map(lambda x: f"{x:.2f}%")

    display_df["CONFIDENCE"] = display_df[
        "CONFIDENCE"
    ].round(2)

    display_df["PRICE"] = display_df[
        "PRICE"
    ].map(lambda x: f"{x:.8g}")

    display_df["RSI"] = display_df[
        "RSI"
    ].round(1)

    for col in [
        "MOM 6H",
        "MOM 12H",
        "MOM 24H"
    ]:

        display_df[col] = display_df[
            col
        ].map(lambda x: f"{x:+.2f}%")

    display_df["VOL Z"] = display_df[
        "VOL Z"
    ].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # TOP SIGNALS
    # ========================================================

    st.divider()

    st.subheader("🎯 Top GPR Signals")

    top_n = min(
        6,
        len(df)
    )

    top_df = df.head(top_n)

    columns = st.columns(2)

    for idx, (_, row) in enumerate(
        top_df.iterrows()
    ):

        with columns[idx % 2]:

            with st.container(border=True):

                st.markdown(
                    f"### {row['COIN']}"
                )

                signal = row["SIGNAL"]

                if signal == "LONG":
                    st.markdown(
                        "🟢 **LONG**"
                    )
                else:
                    st.markdown(
                        "🔴 **SHORT**"
                    )

                a, b = st.columns(2)

                with a:
                    st.metric(
                        "GPR Score",
                        f"{row['GPR SCORE']:.1f}"
                    )

                    st.metric(
                        "GPR Return",
                        f"{row['GPR RETURN']:+.2f}%"
                    )

                with b:
                    st.metric(
                        "Uncertainty",
                        f"{row['UNCERTAINTY']:.2f}%"
                    )

                    st.metric(
                        "Confidence",
                        f"{row['CONFIDENCE']:.2f}"
                    )

    # ========================================================
    # DETAIL ANALYSIS
    # ========================================================

    st.divider()

    st.subheader("🔎 Detail Analysis")

    selected_coin = st.selectbox(
        "Pilih coin",
        df["COIN"].tolist()
    )

    selected = df[
        df["COIN"] == selected_coin
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "SIGNAL",
            selected["SIGNAL"]
        )

    with c2:
        st.metric(
            "GPR SCORE",
            f"{selected['GPR SCORE']:.1f}"
        )

    with c3:
        st.metric(
            "FORECAST",
            f"{selected['GPR RETURN']:+.2f}%"
        )

    with c4:
        st.metric(
            "UNCERTAINTY",
            f"{selected['UNCERTAINTY']:.2f}%"
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Harga",
            f"{selected['PRICE']:.8g}"
        )

        st.metric(
            "Momentum 24H",
            f"{selected['MOM 24H']:+.2f}%"
        )

    with c2:
        st.metric(
            "RSI 14",
            f"{selected['RSI']:.1f}"
        )

        st.metric(
            "Volume Z",
            f"{selected['VOL Z']:.2f}"
        )

    with c3:
        st.metric(
            "Momentum 6H",
            f"{selected['MOM 6H']:+.2f}%"
        )

        st.metric(
            "EMA12 Distance",
            f"{selected['EMA12 DIST']:+.2f}%"
        )

    with c4:
        st.metric(
            "Momentum 12H",
            f"{selected['MOM 12H']:+.2f}%"
        )

        st.metric(
            "EMA24 Distance",
            f"{selected['EMA24 DIST']:+.2f}%"
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
                "Tidak ada error pada coin yang dianalisis."
            )

        else:

            st.dataframe(
                errors_df,
                use_container_width=True,
                hide_index=True
            )

    # ========================================================
    # EXPORT
    # ========================================================

    st.divider()

    csv = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Ranking CSV",
        data=csv,
        file_name=(
            f"gpr_crypto_{active_tf}.csv"
        ),
        mime="text/csv"
    )

else:

    # ========================================================
    # INITIAL STATE
    # ========================================================

    st.info(
        f"Belum ada hasil. Tekan "
        f"**SCAN FUTURES {timeframe}**."
    )

    st.markdown(
        """
        ### Struktur scanner

        **GPR Crypto Futures Scanner**

        - Primary: **1H**
        - Secondary: **15m**
        - Provider: **OKX public market data**
        - API key: **tidak diperlukan**
        - Model: **Gaussian Process Regression**
        - Output: **LONG / SHORT**
        - Ranking: **GPR Score**
        - Forecast: **GPR Return**
        - Risk: **Uncertainty**
        - Confidence: **Prediction / Uncertainty**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GPR Crypto Futures Scanner · USDT Perpetual · "
    "Public Market Data · Research Prototype"
)
