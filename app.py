import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    WhiteKernel,
    ConstantKernel,
)


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
# CONFIG
# ============================================================

OKX_BASE = "https://www.okx.com"

OKX_CANDLE_URL = (
    f"{OKX_BASE}/api/v5/market/candles"
)

# Static universe.
# Sengaja tidak mengambil daftar instrument dari exchangeInfo
# karena endpoint tersebut sebelumnya menghasilkan 403/451.
UNIVERSE = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "BNB-USDT-SWAP",
    "ADA-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "LINK-USDT-SWAP",
    "DOT-USDT-SWAP",
    "TRX-USDT-SWAP",
    "LTC-USDT-SWAP",
    "BCH-USDT-SWAP",
    "SUI-USDT-SWAP",
    "TON-USDT-SWAP",
    "SHIB-USDT-SWAP",
    "PEPE-USDT-SWAP",
    "UNI-USDT-SWAP",
    "NEAR-USDT-SWAP",
    "APT-USDT-SWAP",
    "ICP-USDT-SWAP",
    "ETC-USDT-SWAP",
    "FIL-USDT-SWAP",
    "ATOM-USDT-SWAP",
    "HBAR-USDT-SWAP",
    "AR-USDT-SWAP",
    "OP-USDT-SWAP",
    "ARB-USDT-SWAP",
    "MKR-USDT-SWAP",
    "AAVE-USDT-SWAP",
    "ALGO-USDT-SWAP",
    "XLM-USDT-SWAP",
    "VET-USDT-SWAP",
    "INJ-USDT-SWAP",
    "SEI-USDT-SWAP",
    "TIA-USDT-SWAP",
    "WIF-USDT-SWAP",
    "BONK-USDT-SWAP",
    "JUP-USDT-SWAP",
    "WLD-USDT-SWAP",
    "RUNE-USDT-SWAP",
    "STX-USDT-SWAP",
    "IMX-USDT-SWAP",
    "GRT-USDT-SWAP",
    "SAND-USDT-SWAP",
    "MANA-USDT-SWAP",
    "AXS-USDT-SWAP",
    "GALA-USDT-SWAP",
    "APE-USDT-SWAP",
    "EGLD-USDT-SWAP",
    "FLOW-USDT-SWAP",
    "EOS-USDT-SWAP",
    "XTZ-USDT-SWAP",
    "THETA-USDT-SWAP",
    "KAS-USDT-SWAP",
    "TAO-USDT-SWAP",
    "ENA-USDT-SWAP",
    "ONDO-USDT-SWAP",
    "PENDLE-USDT-SWAP",
    "JASMY-USDT-SWAP",
    "PYTH-USDT-SWAP",
    "W-USDT-SWAP",
    "STRK-USDT-SWAP",
    "NOT-USDT-SWAP",
    "BOME-USDT-SWAP",
    "FLOKI-USDT-SWAP",
    "ORDI-USDT-SWAP",
    "SATS-USDT-SWAP",
    "1000PEPE-USDT-SWAP",
    "1000BONK-USDT-SWAP",
    "1000SHIB-USDT-SWAP",
    "CRV-USDT-SWAP",
    "LDO-USDT-SWAP",
    "COMP-USDT-SWAP",
    "SNX-USDT-SWAP",
    "DYDX-USDT-SWAP",
    "ENS-USDT-SWAP",
    "ZK-USDT-SWAP",
    "ASTR-USDT-SWAP",
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #111827 !important;
    }

    .subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }

    /* ==============================
       DARK METRIC CARD
       ============================== */

    .metric-card {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #CBD5E1;
        font-size: 0.90rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-sub {
        color: #94A3B8;
        font-size: 0.82rem;
        margin-top: 7px;
    }

    /* ==============================
       RANK CARD
       ============================== */

    .rank-card {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .rank-symbol {
        color: #0F172A;
        font-size: 1.15rem;
        font-weight: 800;
    }

    .rank-score {
        color: #0F766E;
        font-size: 1.35rem;
        font-weight: 800;
    }

    .rank-meta {
        color: #475569;
        font-size: 0.86rem;
    }

    /* ==============================
       BUTTON
       ============================== */

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 800 !important;
        min-height: 46px !important;
    }

    /* ==============================
       DATAFRAME
       ============================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #CBD5E1;
        border-radius: 10px;
    }

    /* ==============================
       INFO TEXT
       ============================== */

    .small-note {
        color: #64748B;
        font-size: 0.82rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HTTP SESSION
# ============================================================

@st.cache_resource
def get_session():

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/130 Safari/537.36"
            ),
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
    )

    return session


SESSION = get_session()


# ============================================================
# FETCH OKX CANDLES
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_okx_candles(inst_id, limit=160):

    params = {
        "instId": inst_id,
        "bar": "1H",
        "limit": str(min(int(limit), 300)),
    }

    last_error = None

    for attempt in range(3):

        try:

            response = SESSION.get(
                OKX_CANDLE_URL,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:

                last_error = (
                    f"HTTP {response.status_code}"
                )

                time.sleep(
                    0.7 * (attempt + 1)
                )

                continue

            payload = response.json()

            if payload.get("code") != "0":

                last_error = payload.get(
                    "msg",
                    "OKX API error",
                )

                time.sleep(0.5)

                continue

            rows = payload.get("data", [])

            if not rows:

                last_error = "No candle data"

                continue

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
                    "confirm",
                ],
            )

            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vol_quote",
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

            df = df.dropna(
                subset=["close"]
            )

            df = df.sort_values(
                "ts"
            ).reset_index(drop=True)

            # Buang candle yang sedang berjalan.
            if len(df) >= 3:

                df = df.iloc[:-1].copy()

            if len(df) < 70:

                raise ValueError(
                    f"Only {len(df)} candles"
                )

            return df

        except Exception as error:

            last_error = str(error)

            time.sleep(
                0.6 * (attempt + 1)
            )

    raise RuntimeError(
        last_error or "Unknown OKX error"
    )


# ============================================================
# MATHEMATICAL HELPERS
# ============================================================

def sigmoid(value):

    value = np.clip(
        value,
        -12,
        12,
    )

    return 1.0 / (
        1.0 + np.exp(-value)
    )


# ============================================================
# GPR ENGINE
# ============================================================

def calculate_gpr(
    df,
    horizon=10,
):

    if df is None:

        raise ValueError(
            "Data kosong"
        )

    if len(df) < 70:

        raise ValueError(
            "Candle kurang dari 70"
        )

    close = (
        df["close"]
        .astype(float)
        .values
    )

    volume = (
        df["volume"]
        .astype(float)
        .values
    )

    high = (
        df["high"]
        .astype(float)
        .values
    )

    low = (
        df["low"]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # LOG PRICE
    # --------------------------------------------------------

    log_price = np.log(
        np.maximum(
            close,
            1e-12,
        )
    )

    # Gunakan 80 candle terakhir.
    recent = log_price[-80:]

    n = len(recent)

    x = np.arange(
        n,
        dtype=float,
    ).reshape(-1, 1)

    mean_price = recent.mean()

    std_price = (
        recent.std() + 1e-12
    )

    y = (
        recent - mean_price
    ) / std_price

    # --------------------------------------------------------
    # GPR KERNEL
    # --------------------------------------------------------

    kernel = (
        ConstantKernel(
            1.0,
            constant_value_bounds="fixed",
        )
        *
        RBF(
            length_scale=18.0,
            length_scale_bounds="fixed",
        )
        +
        WhiteKernel(
            noise_level=0.12,
            noise_level_bounds="fixed",
        )
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        normalize_y=False,
        random_state=42,
    )

    model.fit(
        x,
        y,
    )

    # --------------------------------------------------------
    # FORECAST
    # --------------------------------------------------------

    future_x = np.arange(
        n,
        n + int(horizon),
        dtype=float,
    ).reshape(-1, 1)

    prediction, uncertainty = (
        model.predict(
            future_x,
            return_std=True,
        )
    )

    predicted_log_price = (
        prediction * std_price
        + mean_price
    )

    current_log_price = log_price[-1]

    future_return = (
        np.exp(
            predicted_log_price[-1]
            - current_log_price
        )
        - 1.0
    )

    # --------------------------------------------------------
    # GPR SLOPE
    # --------------------------------------------------------

    gpr_slope = float(
        predicted_log_price[-1]
        - predicted_log_price[0]
    )

    direction = (
        "LONG"
        if gpr_slope >= 0
        else "SHORT"
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum_6h = (
        close[-1] / close[-7]
        - 1.0
    )

    momentum_24h = (
        close[-1] / close[-25]
        - 1.0
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    recent_volume = np.mean(
        volume[-8:]
    )

    base_volume = np.mean(
        volume[-40:-8]
    )

    volume_ratio = (
        recent_volume
        / max(
            base_volume,
            1e-12,
        )
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    returns = np.diff(
        log_price[-50:]
    )

    volatility = float(
        np.std(returns)
        * np.sqrt(24)
    )

    average_uncertainty = float(
        np.mean(uncertainty)
    )

    # --------------------------------------------------------
    # DIRECTION STRENGTH
    #
    # Penting:
    # LONG dan SHORT sama-sama bisa mendapatkan score tinggi.
    # --------------------------------------------------------

    direction_strength = sigmoid(
        abs(gpr_slope)
        /
        max(
            volatility * 0.50,
            0.003,
        )
    )

    # --------------------------------------------------------
    # EXPECTED MOVE
    # --------------------------------------------------------

    move_strength = np.clip(
        abs(future_return)
        / 0.08,
        0,
        1,
    )

    # --------------------------------------------------------
    # MOMENTUM STRENGTH
    # --------------------------------------------------------

    momentum_strength = sigmoid(
        (
            abs(momentum_6h)
            +
            0.5 * abs(momentum_24h)
        )
        / 0.025
    )

    # --------------------------------------------------------
    # VOLUME CONFIRMATION
    # --------------------------------------------------------

    volume_strength = np.clip(
        (
            volume_ratio
            - 0.75
        )
        / 1.25,
        0,
        1,
    )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    uncertainty_strength = np.clip(
        1.0
        -
        average_uncertainty
        / 0.12,
        0,
        1,
    )

    # --------------------------------------------------------
    # FINAL GPR SCORE
    # --------------------------------------------------------

    score = 100.0 * (

        0.42 * direction_strength

        +

        0.23 * move_strength

        +

        0.15 * momentum_strength

        +

        0.10 * volume_strength

        +

        0.10 * uncertainty_strength
    )

    score = float(
        np.clip(
            score,
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = 100.0 * np.clip(

        0.55 * direction_strength

        +

        0.25 * uncertainty_strength

        +

        0.20 * volume_strength,

        0,
        1,
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = np.roll(
        close,
        1,
    )

    true_range = np.maximum.reduce(
        [
            high - low,
            np.abs(
                high - previous_close
            ),
            np.abs(
                low - previous_close
            ),
        ]
    )

    atr_percent = (
        np.mean(
            true_range[-14:]
        )
        /
        max(
            close[-1],
            1e-12,
        )
        * 100
    )

    return {

        "score": score,

        "confidence": float(
            np.clip(
                confidence,
                0,
                100,
            )
        ),

        "direction": direction,

        "gpr_return": float(
            future_return * 100
        ),

        "momentum_6h": float(
            momentum_6h * 100
        ),

        "momentum_24h": float(
            momentum_24h * 100
        ),

        "volume_ratio": float(
            volume_ratio
        ),

        "uncertainty": float(
            average_uncertainty * 100
        ),

        "atr_14": float(
            atr_percent
        ),

        "last_price": float(
            close[-1]
        ),
    }


# ============================================================
# ANALYZE ONE COIN
# ============================================================

def analyze_coin(
    instrument,
    horizon,
):

    candles = fetch_okx_candles(
        instrument,
        limit=160,
    )

    result = calculate_gpr(
        candles,
        horizon=horizon,
    )

    result["instrument"] = instrument

    result["symbol"] = (
        instrument
        .replace(
            "-USDT-SWAP",
            "",
        )
    )

    return result


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(
    container,
    label,
    value,
    subtitle="",
):

    container.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-sub">
                {subtitle}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📈 GPR CRYPTO FUTURES SCANNER"
)

st.markdown(
    """
    <div class="subtitle">
    Target utama:
    <b>USDT Perpetual · Timeframe 1H</b>
    · GPR sebagai mesin ranking utama
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Scanner Settings"
    )

    horizon_label = st.selectbox(
        "Horizon GPR",

        [
            "5 candle / 5 jam",
            "10 candle / 10 jam",
            "15 candle / 15 jam",
            "20 candle / 20 jam",
            "24 candle / 24 jam",
        ],

        index=1,
    )

    horizon = int(
        horizon_label.split()[0]
    )

    coin_count = st.selectbox(
        "Jumlah coin",

        [
            25,
            50,
            75,
            79,
        ],

        index=2,
    )

    min_score = st.slider(
        "Minimum score",

        min_value=0,
        max_value=100,

        value=70,

        step=5,
    )

    workers = st.slider(
        "Parallel workers",

        min_value=1,
        max_value=8,

        value=6,

        step=1,
    )

    st.divider()

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
# DATA PROVIDER MESSAGE
# ============================================================

st.info(
    """
    **Mode data:** public market data, tanpa API key.
    Universe futures dibuat statis agar scanner tidak bergantung
    pada endpoint instruments/exchangeInfo yang sebelumnya menghasilkan
    HTTP 403/451.
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:

    st.session_state.results = None

if "errors" not in st.session_state:

    st.session_state.errors = []

if "elapsed" not in st.session_state:

    st.session_state.elapsed = 0.0

if "universe_count" not in st.session_state:

    st.session_state.universe_count = 0


# ============================================================
# SCANNER
# ============================================================

if scan_button:

    selected_universe = (
        UNIVERSE[:coin_count]
    )

    total = len(
        selected_universe
    )

    progress = st.progress(
        0
    )

    status = st.empty()

    results = []

    errors = []

    started = time.time()

    # --------------------------------------------------------
    # PARALLEL FETCH
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        future_map = {

            executor.submit(
                analyze_coin,
                instrument,
                horizon,
            ): instrument

            for instrument
            in selected_universe
        }

        completed = 0

        for future in as_completed(
            future_map
        ):

            instrument = (
                future_map[future]
            )

            completed += 1

            progress.progress(
                completed / total
            )

            status.info(
                f"Scanning "
                f"{completed}/{total} · "
                f"{instrument}"
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
                        "symbol": instrument.replace(
                            "-USDT-SWAP",
                            "",
                        ),
                        "error": str(
                            error
                        ),
                    }
                )

    elapsed = (
        time.time()
        - started
    )

    # --------------------------------------------------------
    # BUILD RESULT DATAFRAME
    # --------------------------------------------------------

    if results:

        result_df = pd.DataFrame(
            results
        )

        result_df = result_df.sort_values(
            [
                "score",
                "confidence",
            ],
            ascending=[
                False,
                False,
            ],
        ).reset_index(
            drop=True
        )

        result_df.insert(
            0,
            "rank",
            np.arange(
                1,
                len(result_df) + 1,
            ),
        )

    else:

        result_df = pd.DataFrame()

    st.session_state.results = (
        result_df
    )

    st.session_state.errors = (
        errors
    )

    st.session_state.elapsed = (
        elapsed
    )

    st.session_state.universe_count = (
        total
    )

    progress.empty()

    status.success(
        f"Scan selesai: "
        f"{len(results)} berhasil, "
        f"{len(errors)} gagal "
        f"dalam {elapsed:.1f} detik."
    )


# ============================================================
# NO RESULT YET
# ============================================================

if st.session_state.results is None:

    st.warning(
        "Belum ada hasil. "
        "Tekan **SCAN FUTURES 1H** di sidebar."
    )

    st.stop()


# ============================================================
# RESULT VARIABLES
# ============================================================

df = st.session_state.results

errors = (
    st.session_state.errors
)

universe_count = (
    st.session_state.universe_count
)

elapsed = (
    st.session_state.elapsed
)

passing = df[
    df["score"] >= min_score
].copy()

candidates = passing[
    passing["confidence"] >= 55
].copy()

if len(df) > 0:

    highest_return = (
        df["gpr_return"].max()
    )

else:

    highest_return = 0.0


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)

metric_card(
    col1,
    "UNIVERSE",
    str(universe_count),
    "USDT Perpetual",
)

metric_card(
    col2,
    "TIMEFRAME",
    "1H",
    "Primary timeframe",
)

metric_card(
    col3,
    "HORIZON",
    f"{horizon}H",
    "GPR forecast",
)

metric_card(
    col4,
    "PROVIDER",
    "OKX",
    "Public market data",
)


st.write("")


# ============================================================
# SECONDARY METRICS
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)

metric_card(
    col1,
    "BERHASIL DIANALISIS",
    str(len(df)),
    f"{len(errors)} gagal",
)

metric_card(
    col2,
    f"LOLOS SCORE ≥ {min_score}",
    str(len(passing)),
    "GPR ranking",
)

metric_card(
    col3,
    "KANDIDAT",
    str(len(candidates)),
    "Score + confidence",
)

metric_card(
    col4,
    "GPR RETURN TERTINGGI",
    f"{highest_return:+.2f}%",
    f"Horizon {horizon}H",
)


st.write("")


# ============================================================
# TABS
# ============================================================

tab_rank, tab_detail, tab_diag = st.tabs(
    [
        "🏆 RANKING",
        "📈 DETAIL",
        "🧾 ERROR / DIAGNOSTIC",
    ]
)


# ============================================================
# RANKING TAB
# ============================================================

with tab_rank:

    st.subheader(
        "🏆 GPR Futures Ranking"
    )

    if len(passing) == 0:

        st.warning(
            f"Belum ada coin yang mencapai "
            f"minimum score {min_score}. "
            f"Turunkan threshold untuk melihat "
            f"lebih banyak kandidat."
        )

    else:

        display_df = passing.copy()

        display_df["Score"] = (
            display_df["score"]
            .map(
                lambda x: f"{x:.1f}"
            )
        )

        display_df["GPR Return"] = (
            display_df["gpr_return"]
            .map(
                lambda x: f"{x:+.2f}%"
            )
        )

        display_df["Confidence"] = (
            display_df["confidence"]
            .map(
                lambda x: f"{x:.1f}%"
            )
        )

        display_df["Momentum 6H"] = (
            display_df["momentum_6h"]
            .map(
                lambda x: f"{x:+.2f}%"
            )
        )

        display_df["Momentum 24H"] = (
            display_df["momentum_24h"]
            .map(
                lambda x: f"{x:+.2f}%"
            )
        )

        display_df["Volume"] = (
            display_df["volume_ratio"]
            .map(
                lambda x: f"{x:.2f}x"
            )
        )

        display_df["Uncertainty"] = (
            display_df["uncertainty"]
            .map(
                lambda x: f"{x:.2f}%"
            )
        )

        display_df = display_df[
            [
                "rank",
                "symbol",
                "direction",
                "Score",
                "GPR Return",
                "Confidence",
                "Momentum 6H",
                "Momentum 24H",
                "Volume",
                "Uncertainty",
            ]
        ]

        display_df = display_df.rename(
            columns={
                "rank": "Rank",
                "symbol": "Coin",
                "direction": "Signal",
            }
        )

        st.dataframe(
            display_df,

            use_container_width=True,

            hide_index=True,

            height=min(
                700,
                120
                +
                len(display_df) * 38,
            ),
        )

        st.caption(
            """
            **Catatan:** Score adalah ranking strength
            model, bukan probabilitas profit.
            GPR Return adalah estimasi pergerakan model
            untuk horizon yang dipilih.
            """
        )


# ============================================================
# DETAIL TAB
# ============================================================

with tab_detail:

    st.subheader(
        "📈 Detail Kandidat"
    )

    if len(candidates) == 0:

        st.info(
            "Belum ada kandidat dengan "
            "score dan confidence yang memenuhi threshold."
        )

    else:

        for _, row in (
            candidates.head(10).iterrows()
        ):

            if row["direction"] == "LONG":

                signal = "🟢 LONG"

            else:

                signal = "🔴 SHORT"

            st.markdown(
                f"""
                <div class="rank-card">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        gap:20px;
                        align-items:center;
                    ">

                        <div>

                            <div class="rank-symbol">
                                #{int(row["rank"])}
                                &nbsp;
                                {row["symbol"]}
                            </div>

                            <div class="rank-meta">
                                {signal}
                                · Momentum 6H:
                                {row["momentum_6h"]:+.2f}%
                                · Volume:
                                {row["volume_ratio"]:.2f}x
                            </div>

                        </div>

                        <div style="
                            text-align:right;
                        ">

                            <div class="rank-score">
                                Score
                                {row["score"]:.1f}
                            </div>

                            <div class="rank-meta">
                                GPR:
                                {row["gpr_return"]:+.2f}%
                                · Confidence:
                                {row["confidence"]:.1f}%
                            </div>

                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DIAGNOSTIC TAB
# ============================================================

with tab_diag:

    st.subheader(
        "🧾 Error / Diagnostic"
    )

    st.write(
        f"""
        Universe: **{universe_count}**
        · Berhasil: **{len(df)}**
        · Gagal: **{len(errors)}**
        · Waktu: **{elapsed:.1f}s**
        """
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
            "Tidak ada error pada scan terakhir."
        )

    st.info(
        """
        Coin yang gagal tidak otomatis berarti model gagal.
        Penyebab yang umum adalah kontrak tidak tersedia
        pada provider, candle kosong, atau pembatasan
        request sementara.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    GPR Crypto Futures Scanner ·
    USDT Perpetual · 1H ·
    Public Market Data ·
    Research Prototype
    """
)
