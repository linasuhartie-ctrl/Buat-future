import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ============================================================
# GPR CRYPTO FUTURES SCANNER
# Target utama:
#   USDT Perpetual
#   Timeframe 1H
#   GPR = ranking utama
#
# Public market data - NO API KEY
# Provider utama: OKX
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

.main-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 0px;
}

.subtitle {
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 20px;
}

.card {
    border-radius: 14px;
    padding: 18px 20px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    min-height: 125px;
}

.card-label {
    font-size: 14px;
    color: #6b7280;
    font-weight: 600;
    margin-bottom: 8px;
}

.card-value {
    font-size: 30px;
    font-weight: 800;
    color: #111827;
}

.card-sub {
    font-size: 13px;
    color: #6b7280;
    margin-top: 5px;
}

.signal-long {
    color: #059669;
    font-weight: 800;
}

.signal-short {
    color: #dc2626;
    font-weight: 800;
}

.signal-watch {
    color: #d97706;
    font-weight: 800;
}

.small-note {
    font-size: 12px;
    color: #6b7280;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

OKX_BASE = "https://www.okx.com/api/v5"

# Static fallback universe.
# Tujuannya agar scanner TIDAK bergantung pada endpoint
# instruments yang sebelumnya menghasilkan 403/451.
STATIC_SYMBOLS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "ADA-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "LINK-USDT-SWAP",
    "DOT-USDT-SWAP",
    "LTC-USDT-SWAP",
    "BCH-USDT-SWAP",
    "UNI-USDT-SWAP",
    "ATOM-USDT-SWAP",
    "ETC-USDT-SWAP",
    "FIL-USDT-SWAP",
    "APT-USDT-SWAP",
    "ARB-USDT-SWAP",
    "OP-USDT-SWAP",
    "SUI-USDT-SWAP",
    "SEI-USDT-SWAP",
    "NEAR-USDT-SWAP",
    "INJ-USDT-SWAP",
    "AAVE-USDT-SWAP",
    "MKR-USDT-SWAP",
    "CRV-USDT-SWAP",
    "LDO-USDT-SWAP",
    "IMX-USDT-SWAP",
    "GRT-USDT-SWAP",
    "ALGO-USDT-SWAP",
    "XLM-USDT-SWAP",
    "TRX-USDT-SWAP",
    "TON-USDT-SWAP",
    "HBAR-USDT-SWAP",
    "ICP-USDT-SWAP",
    "VET-USDT-SWAP",
    "SAND-USDT-SWAP",
    "MANA-USDT-SWAP",
    "AXS-USDT-SWAP",
    "GALA-USDT-SWAP",
    "APE-USDT-SWAP",
    "EOS-USDT-SWAP",
    "THETA-USDT-SWAP",
    "RUNE-USDT-SWAP",
    "WLD-USDT-SWAP",
    "TIA-USDT-SWAP",
    "JUP-USDT-SWAP",
    "PYTH-USDT-SWAP",
    "PEPE-USDT-SWAP",
    "SHIB-USDT-SWAP",
    "BONK-USDT-SWAP",
    "FLOKI-USDT-SWAP",
    "WIF-USDT-SWAP",
    "ORDI-USDT-SWAP",
    "STX-USDT-SWAP",
    "ENA-USDT-SWAP",
    "ONDO-USDT-SWAP",
    "NOT-USDT-SWAP",
    "ZRO-USDT-SWAP",
    "JTO-USDT-SWAP",
    "TAO-USDT-SWAP",
    "KAS-USDT-SWAP",
    "FET-USDT-SWAP",
    "RENDER-USDT-SWAP",
    "AR-USDT-SWAP",
    "DYM-USDT-SWAP",
    "STRK-USDT-SWAP",
    "MANTA-USDT-SWAP",
    "BLUR-USDT-SWAP",
    "MINA-USDT-SWAP",
    "FLOW-USDT-SWAP",
    "QNT-USDT-SWAP",
    "XTZ-USDT-SWAP",
    "ZEC-USDT-SWAP",
    "DASH-USDT-SWAP",
    "COMP-USDT-SWAP",
    "SNX-USDT-SWAP",
    "1INCH-USDT-SWAP",
    "YFI-USDT-SWAP",
    "SUSHI-USDT-SWAP",
    "ENS-USDT-SWAP",
    "DYDX-USDT-SWAP",
    "WOO-USDT-SWAP",
    "JASMY-USDT-SWAP",
    "CFX-USDT-SWAP",
    "CKB-USDT-SWAP",
    "IOTA-USDT-SWAP",
    "AXL-USDT-SWAP",
    "APE-USDT-SWAP",
    "MAVIA-USDT-SWAP",
    "TURBO-USDT-SWAP",
]


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_session():
    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/130.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Connection": "keep-alive",
    })

    return session


SESSION = get_session()


# ============================================================
# OKX API
# ============================================================

def okx_get(path, params=None, timeout=12):
    url = OKX_BASE + path

    try:
        r = SESSION.get(
            url,
            params=params,
            timeout=timeout
        )

        r.raise_for_status()

        payload = r.json()

        if payload.get("code") != "0":
            raise RuntimeError(
                f"OKX code={payload.get('code')} "
                f"msg={payload.get('msg')}"
            )

        return payload.get("data", [])

    except Exception as e:
        raise RuntimeError(str(e))


# ============================================================
# GET STATIC FUTURES UNIVERSE
# ============================================================

@st.cache_data(ttl=300)
def get_universe():

    # Kita sengaja tidak memanggil:
    # /public/instruments
    #
    # karena endpoint tersebut pada deployment sebelumnya
    # terkena HTTP 403/451.
    #
    # Static universe membuat scanner tetap dapat bekerja
    # selama endpoint candles OKX dapat diakses.

    return STATIC_SYMBOLS.copy()


# ============================================================
# GET 1H CANDLES
# ============================================================

def get_candles(inst_id, limit=180):

    data = okx_get(
        "/market/candles",
        params={
            "instId": inst_id,
            "bar": "1H",
            "limit": min(limit, 300)
        },
        timeout=12
    )

    if not data:
        raise RuntimeError("No candle data")

    rows = []

    for x in data:

        # OKX candle:
        # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]

        if len(x) < 6:
            continue

        rows.append({
            "ts": pd.to_datetime(
                int(x[0]),
                unit="ms",
                utc=True
            ),
            "open": float(x[1]),
            "high": float(x[2]),
            "low": float(x[3]),
            "close": float(x[4]),
            "volume": float(x[5]),
            "confirm": x[8] if len(x) > 8 else "1"
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Empty dataframe")

    df = df.sort_values("ts").reset_index(drop=True)

    # Buang candle terakhir jika belum confirmed.
    if "confirm" in df.columns:
        df = df[df["confirm"].astype(str) == "1"].copy()

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(df):

    d = df.copy()

    close = d["close"]
    high = d["high"]
    low = d["low"]
    volume = d["volume"]

    # Return
    d["ret1"] = close.pct_change(1)
    d["ret3"] = close.pct_change(3)
    d["ret6"] = close.pct_change(6)
    d["ret12"] = close.pct_change(12)
    d["ret24"] = close.pct_change(24)

    # Momentum
    d["mom6"] = close / close.shift(6) - 1
    d["mom12"] = close / close.shift(12) - 1
    d["mom24"] = close / close.shift(24) - 1

    # Volatility
    d["vol6"] = d["ret1"].rolling(6).std()
    d["vol12"] = d["ret1"].rolling(12).std()
    d["vol24"] = d["ret1"].rolling(24).std()

    # Range
    d["range"] = (high - low) / close
    d["range6"] = d["range"].rolling(6).mean()

    # Candle body
    d["body"] = (close - d["open"]) / d["open"]

    # Trend position
    d["ema12"] = close.ewm(span=12, adjust=False).mean()
    d["ema24"] = close.ewm(span=24, adjust=False).mean()
    d["ema48"] = close.ewm(span=48, adjust=False).mean()

    d["ema12_dist"] = close / d["ema12"] - 1
    d["ema24_dist"] = close / d["ema24"] - 1
    d["ema48_dist"] = close / d["ema48"] - 1

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    d["rsi14"] = 100 - (100 / (1 + rs))

    # Volume
    vol_mean = volume.rolling(24).mean()
    vol_std = volume.rolling(24).std()

    d["volume_z"] = (
        (volume - vol_mean) /
        vol_std.replace(0, np.nan)
    )

    return d


# ============================================================
# GPR MODEL
# ============================================================

FEATURES = [
    "ret1",
    "ret3",
    "ret6",
    "ret12",
    "ret24",
    "mom6",
    "mom12",
    "mom24",
    "vol6",
    "vol12",
    "vol24",
    "range6",
    "body",
    "ema12_dist",
    "ema24_dist",
    "ema48_dist",
    "rsi14",
    "volume_z",
]


def calculate_gpr(df, horizon):

    d = make_features(df)

    # Future return = target
    d["future_return"] = (
        d["close"].shift(-horizon) /
        d["close"] - 1
    )

    model_df = d[FEATURES + ["future_return"]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    # Minimal data requirement
    if len(model_df) < 80:
        raise RuntimeError(
            f"Data training kurang: {len(model_df)}"
        )

    # Gunakan data terbaru agar model fokus pada regime sekarang.
    model_df = model_df.tail(140)

    X = model_df[FEATURES].values
    y = model_df["future_return"].values

    # Winsorize target untuk mengurangi efek outlier ekstrem.
    low_q = np.quantile(y, 0.05)
    high_q = np.quantile(y, 0.95)

    y = np.clip(y, low_q, high_q)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(
            1.0,
            (0.1, 10.0)
        )
        * RBF(
            length_scale=1.0,
            length_scale_bounds=(0.1, 10.0)
        )
        + WhiteKernel(
            noise_level=0.05,
            noise_level_bounds=(1e-5, 1.0)
        )
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=42
    )

    gpr.fit(X_scaled, y)

    # Latest row
    latest = d[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if latest.empty:
        raise RuntimeError("Latest features kosong")

    x_now = latest[FEATURES].iloc[-1].values.reshape(1, -1)

    x_now_scaled = scaler.transform(x_now)

    pred, std = gpr.predict(
        x_now_scaled,
        return_std=True
    )

    pred_return = float(pred[0])
    uncertainty = float(std[0])

    return {
        "prediction": pred_return,
        "uncertainty": uncertainty,
        "rsi": float(d["rsi14"].iloc[-1]),
        "mom6": float(d["mom6"].iloc[-1]),
        "mom12": float(d["mom12"].iloc[-1]),
        "mom24": float(d["mom24"].iloc[-1]),
        "vol24": float(d["vol24"].iloc[-1]),
        "volume_z": float(d["volume_z"].iloc[-1]),
        "ema12_dist": float(d["ema12_dist"].iloc[-1]),
        "ema24_dist": float(d["ema24_dist"].iloc[-1]),
        "last_price": float(d["close"].iloc[-1]),
        "last_time": d["ts"].iloc[-1],
    }


# ============================================================
# GPR SCORE
# ============================================================

def calculate_score(result):

    pred = result["prediction"]
    uncertainty = result["uncertainty"]

    rsi = result["rsi"]
    mom6 = result["mom6"]
    mom12 = result["mom12"]
    ema12 = result["ema12_dist"]
    ema24 = result["ema24_dist"]
    volume_z = result["volume_z"]

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if pred > 0:
        direction = "LONG"
    elif pred < 0:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # --------------------------------------------------------
    # Signal quality
    # --------------------------------------------------------

    # Signal strength
    signal_strength = min(
        abs(pred) / 0.05,
        1.0
    )

    # Confidence berdasarkan prediction / uncertainty.
    confidence_ratio = (
        abs(pred) /
        max(uncertainty, 1e-8)
    )

    confidence = min(
        confidence_ratio / 3.0,
        1.0
    )

    # Momentum agreement
    if direction == "LONG":
        momentum_score = np.mean([
            mom6 > 0,
            mom12 > 0,
            ema12 > 0,
            ema24 > 0
        ])
    elif direction == "SHORT":
        momentum_score = np.mean([
            mom6 < 0,
            mom12 < 0,
            ema12 < 0,
            ema24 < 0
        ])
    else:
        momentum_score = 0

    # RSI quality
    if direction == "LONG":
        rsi_score = (
            1.0 if 45 <= rsi <= 68
            else 0.5 if 35 <= rsi <= 75
            else 0.2
        )

    elif direction == "SHORT":
        rsi_score = (
            1.0 if 32 <= rsi <= 55
            else 0.5 if 25 <= rsi <= 65
            else 0.2
        )

    else:
        rsi_score = 0

    # Volume confirmation
    volume_score = min(
        max((volume_z + 1) / 3, 0),
        1
    )

    # --------------------------------------------------------
    # FINAL GPR SCORE
    # --------------------------------------------------------

    score = (
        45 * signal_strength +
        25 * confidence +
        20 * momentum_score +
        7 * rsi_score +
        3 * volume_score
    )

    return float(
        np.clip(score, 0, 100)
    ), direction


# ============================================================
# ANALYZE ONE SYMBOL
# ============================================================

def analyze_symbol(inst_id, horizon):

    try:

        df = get_candles(
            inst_id,
            limit=180
        )

        result = calculate_gpr(
            df,
            horizon
        )

        score, direction = calculate_score(
            result
        )

        symbol = inst_id.replace(
            "-USDT-SWAP",
            ""
        )

        return {
            "Symbol": symbol,
            "Instrument": inst_id,
            "Signal": direction,
            "GPR Score": round(score, 1),
            "GPR Return": result["prediction"] * 100,
            "Uncertainty": result["uncertainty"] * 100,
            "Confidence": (
                abs(result["prediction"]) /
                max(result["uncertainty"], 1e-8)
            ),
            "Price": result["last_price"],
            "RSI": result["rsi"],
            "Mom 6H": result["mom6"] * 100,
            "Mom 12H": result["mom12"] * 100,
            "Mom 24H": result["mom24"] * 100,
            "Volume Z": result["volume_z"],
            "EMA12 Dist": result["ema12_dist"] * 100,
            "EMA24 Dist": result["ema24_dist"] * 100,
            "Last Candle": result["last_time"],
            "Status": "OK",
        }

    except Exception as e:

        return {
            "Symbol": inst_id.replace(
                "-USDT-SWAP",
                ""
            ),
            "Instrument": inst_id,
            "Signal": "-",
            "GPR Score": np.nan,
            "GPR Return": np.nan,
            "Uncertainty": np.nan,
            "Confidence": np.nan,
            "Price": np.nan,
            "RSI": np.nan,
            "Mom 6H": np.nan,
            "Mom 12H": np.nan,
            "Mom 24H": np.nan,
            "Volume Z": np.nan,
            "EMA12 Dist": np.nan,
            "EMA24 Dist": np.nan,
            "Last Candle": pd.NaT,
            "Status": str(e),
        }


# ============================================================
# SCAN ALL
# ============================================================

def run_scan(symbols, horizon, workers):

    results = []

    progress = st.progress(0)

    status_text = st.empty()

    total = len(symbols)

    completed = 0

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                analyze_symbol,
                symbol,
                horizon
            ): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):

            try:
                results.append(
                    future.result()
                )

            except Exception as e:

                symbol = futures[future]

                results.append({
                    "Symbol": symbol,
                    "Status": str(e)
                })

            completed += 1

            progress.progress(
                completed / total
            )

            status_text.write(
                f"Scanning {completed}/{total}..."
            )

    progress.empty()
    status_text.empty()

    return pd.DataFrame(results)


# ============================================================
# SESSION STATE
# ============================================================

if "scan_result" not in st.session_state:
    st.session_state.scan_result = None

if "scan_time" not in st.session_state:
    st.session_state.scan_time = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Scanner Settings")

horizon = st.sidebar.selectbox(
    "Horizon GPR",
    options=[3, 6, 10, 12, 18, 24],
    index=2,
    format_func=lambda x: f"{x} candle / {x} jam"
)

coin_count = st.sidebar.selectbox(
    "Jumlah coin",
    options=[20, 30, 50, 75, 79],
    index=2
)

minimum_score = st.sidebar.slider(
    "Minimum score",
    min_value=0,
    max_value=100,
    value=70,
    step=5
)

workers = st.sidebar.slider(
    "Parallel workers",
    min_value=1,
    max_value=8,
    value=5
)

st.sidebar.markdown("---")

scan_button = st.sidebar.button(
    "🚀 SCAN FUTURES 1H",
    type="primary",
    use_container_width=True
)

clear_button = st.sidebar.button(
    "🗑️ CLEAR CACHE",
    use_container_width=True
)

if clear_button:

    st.cache_data.clear()

    st.session_state.scan_result = None

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📈 GPR CRYPTO FUTURES SCANNER</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Target utama: USDT Perpetual · Timeframe 1H · '
    'GPR sebagai mesin ranking utama'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INFO
# ============================================================

st.info(
    "Mode data: public market data, tanpa API key. "
    "Universe futures menggunakan daftar static agar scanner "
    "tidak bergantung pada endpoint instruments/exchangeInfo "
    "yang sebelumnya menghasilkan HTTP 403/451."
)


# ============================================================
# SCAN
# ============================================================

if scan_button:

    universe = get_universe()

    selected_symbols = universe[:coin_count]

    start_time = time.time()

    df_result = run_scan(
        selected_symbols,
        horizon,
        workers
    )

    elapsed = time.time() - start_time

    st.session_state.scan_result = df_result
    st.session_state.scan_time = elapsed


# ============================================================
# DISPLAY RESULT
# ============================================================

if st.session_state.scan_result is None:

    st.warning(
        "Belum ada hasil. Tekan **SCAN FUTURES 1H**."
    )

else:

    df = st.session_state.scan_result.copy()

    # --------------------------------------------------------
    # VALID / FAILED
    # --------------------------------------------------------

    success_df = df[
        df["Status"] == "OK"
    ].copy()

    failed_df = df[
        df["Status"] != "OK"
    ].copy()

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    ranked = success_df.copy()

    if not ranked.empty:

        ranked = ranked.sort_values(
            "GPR Score",
            ascending=False
        ).reset_index(drop=True)

    candidates = ranked[
        ranked["GPR Score"] >= minimum_score
    ].copy()

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    universe_count = len(df)
    success_count = len(success_df)
    failed_count = len(failed_df)
    candidate_count = len(candidates)

    if not ranked.empty:
        top_return = ranked.iloc[0]["GPR Return"]
    else:
        top_return = np.nan

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "UNIVERSE",
            universe_count,
            "USDT Perpetual"
        )

    with c2:
        st.metric(
            "TIMEFRAME",
            "1H",
            "Primary timeframe"
        )

    with c3:
        st.metric(
            "HORIZON",
            f"{horizon}H",
            "GPR forecast"
        )

    with c4:
        st.metric(
            "PROVIDER",
            "OKX",
            "Public market data"
        )

    st.markdown("---")

    st.success(
        f"Scan selesai: {success_count} berhasil, "
        f"{failed_count} gagal dalam "
        f"{st.session_state.scan_time:.1f} detik."
    )

    # --------------------------------------------------------
    # SECOND METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "BERHASIL DIANALISIS",
            success_count
        )

    with c2:
        st.metric(
            f"LOLOS SCORE ≥ {minimum_score}",
            candidate_count
        )

    with c3:
        st.metric(
            "KANDIDAT",
            candidate_count
        )

    with c4:

        if np.isfinite(top_return):

            st.metric(
                "GPR RETURN TERTINGGI",
                f"{top_return:+.2f}%",
                f"Horizon {horizon}H"
            )

        else:

            st.metric(
                "GPR RETURN TERTINGGI",
                "-"
            )

    st.markdown("---")

    # ========================================================
    # RANKING
    # ========================================================

    st.subheader("🏆 GPR Futures Ranking")

    if ranked.empty:

        st.warning(
            "Tidak ada coin yang berhasil dianalisis."
        )

    else:

        display_df = ranked.copy()

        # Hanya tampilkan coin sesuai threshold
        display_df = display_df[
            display_df["GPR Score"] >= minimum_score
        ].copy()

        if display_df.empty:

            st.warning(
                f"Belum ada coin yang mencapai minimum score "
                f"{minimum_score}. "
                f"Turunkan threshold untuk melihat kandidat."
            )

        else:

            display_df.insert(
                0,
                "Rank",
                range(1, len(display_df) + 1)
            )

            display_df["GPR Return"] = (
                display_df["GPR Return"]
                .map(lambda x: f"{x:+.2f}%")
            )

            display_df["Uncertainty"] = (
                display_df["Uncertainty"]
                .map(lambda x: f"{x:.2f}%")
            )

            display_df["Confidence"] = (
                display_df["Confidence"]
                .map(lambda x: f"{x:.2f}")
            )

            display_df["Price"] = (
                display_df["Price"]
                .map(lambda x: f"{x:,.6g}")
            )

            display_df["RSI"] = (
                display_df["RSI"]
                .map(lambda x: f"{x:.1f}")
            )

            display_df["Mom 6H"] = (
                display_df["Mom 6H"]
                .map(lambda x: f"{x:+.2f}%")
            )

            display_df["Mom 12H"] = (
                display_df["Mom 12H"]
                .map(lambda x: f"{x:+.2f}%")
            )

            display_df["Mom 24H"] = (
                display_df["Mom 24H"]
                .map(lambda x: f"{x:+.2f}%")
            )

            display_df["Volume Z"] = (
                display_df["Volume Z"]
                .map(lambda x: f"{x:+.2f}")
            )

            display_df["EMA12 Dist"] = (
                display_df["EMA12 Dist"]
                .map(lambda x: f"{x:+.2f}%")
            )

            display_df["EMA24 Dist"] = (
                display_df["EMA24 Dist"]
                .map(lambda x: f"{x:+.2f}%")
            )

            # Hanya kolom penting untuk ranking
            ranking_columns = [
                "Rank",
                "Symbol",
                "Signal",
                "GPR Score",
                "GPR Return",
                "Uncertainty",
                "Confidence",
                "Price",
                "RSI",
                "Mom 6H",
                "Mom 12H",
                "Mom 24H",
                "Volume Z",
            ]

            st.dataframe(
                display_df[ranking_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "🏆 Rank",
                        width="small"
                    ),
                    "Symbol": st.column_config.TextColumn(
                        "COIN"
                    ),
                    "Signal": st.column_config.TextColumn(
                        "SIGNAL"
                    ),
                    "GPR Score": st.column_config.NumberColumn(
                        "GPR SCORE",
                        format="%.1f"
                    ),
                    "GPR Return": st.column_config.TextColumn(
                        "GPR RETURN"
                    ),
                    "Uncertainty": st.column_config.TextColumn(
                        "UNCERTAINTY"
                    ),
                    "Confidence": st.column_config.TextColumn(
                        "CONFIDENCE"
                    ),
                    "Price": st.column_config.TextColumn(
                        "PRICE"
                    ),
                    "RSI": st.column_config.TextColumn(
                        "RSI"
                    ),
                    "Mom 6H": st.column_config.TextColumn(
                        "MOM 6H"
                    ),
                    "Mom 12H": st.column_config.TextColumn(
                        "MOM 12H"
                    ),
                    "Mom 24H": st.column_config.TextColumn(
                        "MOM 24H"
                    ),
                    "Volume Z": st.column_config.TextColumn(
                        "VOL Z"
                    ),
                }
            )

    # ========================================================
    # TOP SIGNAL
    # ========================================================

    if not candidates.empty:

        st.markdown("---")

        st.subheader("🎯 Top GPR Signals")

        top_n = min(
            5,
            len(candidates)
        )

        top = candidates.head(top_n)

        cols = st.columns(top_n)

        for col, (_, row) in zip(
            cols,
            top.iterrows()
        ):

            with col:

                signal = row["Signal"]

                if signal == "LONG":
                    signal_text = "🟢 LONG"
                elif signal == "SHORT":
                    signal_text = "🔴 SHORT"
                else:
                    signal_text = "🟡 NEUTRAL"

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-label">
                            {row["Symbol"]}
                        </div>

                        <div class="card-value">
                            {row["GPR Score"]:.1f}
                        </div>

                        <div class="card-sub">
                            {signal_text}
                        </div>

                        <div class="card-sub">
                            GPR Return:
                            {row["GPR Return"]:+.2f}%
                        </div>

                        <div class="card-sub">
                            Uncertainty:
                            {row["Uncertainty"]:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ========================================================
    # DETAIL
    # ========================================================

    st.markdown("---")

    st.subheader("🔎 Detail Analysis")

    if not ranked.empty:

        selected_symbol = st.selectbox(
            "Pilih coin",
            ranked["Symbol"].tolist()
        )

        detail = ranked[
            ranked["Symbol"] == selected_symbol
        ].iloc[0]

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.metric(
                "Signal",
                detail["Signal"]
            )

        with d2:
            st.metric(
                "GPR Score",
                f'{detail["GPR Score"]:.1f}'
            )

        with d3:
            st.metric(
                "Forecast",
                f'{detail["GPR Return"]:+.2f}%'
            )

        with d4:
            st.metric(
                "Uncertainty",
                f'{detail["Uncertainty"]:.2f}%'
            )

        st.write(
            f"**Harga:** {detail['Price']:,.6g}"
        )

        st.write(
            f"**RSI 14:** {detail['RSI']:.1f}"
        )

        st.write(
            f"**Momentum 6H:** {detail['Mom 6H']:+.2f}%"
        )

        st.write(
            f"**Momentum 12H:** {detail['Mom 12H']:+.2f}%"
        )

        st.write(
            f"**Momentum 24H:** {detail['Mom 24H']:+.2f}%"
        )

        st.write(
            f"**Volume Z:** {detail['Volume Z']:+.2f}"
        )

        st.write(
            f"**EMA12 distance:** {detail['EMA12 Dist']:+.2f}%"
        )

        st.write(
            f"**EMA24 distance:** {detail['EMA24 Dist']:+.2f}%"
        )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    st.markdown("---")

    with st.expander("📋 ERROR / DIAGNOSTIC"):

        if failed_df.empty:

            st.success(
                "Tidak ada error pada coin yang discan."
            )

        else:

            error_display = failed_df[
                ["Symbol", "Status"]
            ].copy()

            st.dataframe(
                error_display,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "GPR Crypto Futures Scanner · "
    "USDT Perpetual · 1H · Public Market Data · "
    "Research Prototype"
)
