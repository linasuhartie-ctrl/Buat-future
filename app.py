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
#
# TARGET UTAMA
#   USDT PERPETUAL
#   TIMEFRAME 1H
#   GPR = MESIN RANKING UTAMA
#
# PUBLIC MARKET DATA - NO API KEY
# PROVIDER UTAMA = OKX
# ============================================================


st.set_page_config(
    page_title="GPR Crypto Futures Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
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

.section-title {
    font-size: 24px;
    font-weight: 750;
    margin-top: 10px;
    margin-bottom: 10px;
}

.small-note {
    color: #6b7280;
    font-size: 12px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

OKX_BASE = "https://www.okx.com/api/v5"


# ============================================================
# STATIC FUTURES UNIVERSE
#
# Tidak menggunakan endpoint instruments/exchangeInfo
# agar tidak terkena 403/451 seperti deployment sebelumnya.
# ============================================================

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
    "MAVIA-USDT-SWAP",
    "TURBO-USDT-SWAP",
]


# ============================================================
# HTTP
# ============================================================

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


# ============================================================
# OKX API
# ============================================================

def okx_get(path, params=None, timeout=15):

    url = OKX_BASE + path

    session = get_session()

    try:

        response = session.get(
            url,
            params=params,
            timeout=timeout
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("code") != "0":

            raise RuntimeError(
                f"OKX code={payload.get('code')} "
                f"msg={payload.get('msg')}"
            )

        return payload.get("data", [])

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"HTTP error: {str(e)}"
        )

    except ValueError:

        raise RuntimeError(
            "Response OKX bukan JSON"
        )

    except Exception as e:

        raise RuntimeError(
            str(e)
        )


# ============================================================
# UNIVERSE
# ============================================================

@st.cache_data(ttl=300)
def get_universe():

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
        timeout=15
    )

    if not data:

        raise RuntimeError(
            "No candle data"
        )

    rows = []

    for x in data:

        if len(x) < 6:
            continue

        try:

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

                "confirm": (
                    x[8]
                    if len(x) > 8
                    else "1"
                )
            })

        except Exception:
            continue

    df = pd.DataFrame(rows)

    if df.empty:

        raise RuntimeError(
            "Empty dataframe"
        )

    df = (
        df
        .sort_values("ts")
        .reset_index(drop=True)
    )

    # Buang candle yang belum confirmed
    if "confirm" in df.columns:

        df = df[
            df["confirm"]
            .astype(str)
            .eq("1")
        ].copy()

    if len(df) < 100:

        raise RuntimeError(
            f"Candle kurang: {len(df)}"
        )

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


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    d["ret1"] = close.pct_change(1)
    d["ret3"] = close.pct_change(3)
    d["ret6"] = close.pct_change(6)
    d["ret12"] = close.pct_change(12)
    d["ret24"] = close.pct_change(24)


    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    d["mom6"] = (
        close / close.shift(6) - 1
    )

    d["mom12"] = (
        close / close.shift(12) - 1
    )

    d["mom24"] = (
        close / close.shift(24) - 1
    )


    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    d["vol6"] = (
        d["ret1"]
        .rolling(6)
        .std()
    )

    d["vol12"] = (
        d["ret1"]
        .rolling(12)
        .std()
    )

    d["vol24"] = (
        d["ret1"]
        .rolling(24)
        .std()
    )


    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    d["range"] = (
        (high - low) / close
    )

    d["range6"] = (
        d["range"]
        .rolling(6)
        .mean()
    )


    # --------------------------------------------------------
    # CANDLE BODY
    # --------------------------------------------------------

    d["body"] = (
        (close - d["open"])
        / d["open"]
    )


    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    d["ema12"] = (
        close
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    d["ema24"] = (
        close
        .ewm(
            span=24,
            adjust=False
        )
        .mean()
    )

    d["ema48"] = (
        close
        .ewm(
            span=48,
            adjust=False
        )
        .mean()
    )


    d["ema12_dist"] = (
        close / d["ema12"] - 1
    )

    d["ema24_dist"] = (
        close / d["ema24"] - 1
    )

    d["ema48_dist"] = (
        close / d["ema48"] - 1
    )


    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(14)
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    d["rsi14"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )


    # --------------------------------------------------------
    # VOLUME Z-SCORE
    # --------------------------------------------------------

    vol_mean = (
        volume
        .rolling(24)
        .mean()
    )

    vol_std = (
        volume
        .rolling(24)
        .std()
    )

    d["volume_z"] = (
        (volume - vol_mean)
        / vol_std.replace(
            0,
            np.nan
        )
    )


    return d


# ============================================================
# GPR FEATURES
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


# ============================================================
# GPR MODEL
# ============================================================

def calculate_gpr(df, horizon):

    d = make_features(df)


    # --------------------------------------------------------
    # TARGET
    #
    # Future return setelah N candle.
    #
    # Contoh horizon = 10
    # berarti prediksi return 10 jam ke depan
    # karena timeframe utama = 1H.
    # --------------------------------------------------------

    d["future_return"] = (
        d["close"].shift(-horizon)
        / d["close"]
        - 1
    )


    model_df = (
        d[
            FEATURES
            + ["future_return"]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    # --------------------------------------------------------
    # MINIMUM DATA
    # --------------------------------------------------------

    if len(model_df) < 80:

        raise RuntimeError(
            f"Data training kurang: "
            f"{len(model_df)}"
        )


    # Fokus ke regime terbaru

    model_df = (
        model_df
        .tail(140)
    )


    X = (
        model_df[FEATURES]
        .values
    )

    y = (
        model_df["future_return"]
        .values
    )


    # --------------------------------------------------------
    # WINSORIZE TARGET
    # --------------------------------------------------------

    low_q = np.quantile(
        y,
        0.05
    )

    high_q = np.quantile(
        y,
        0.95
    )

    y = np.clip(
        y,
        low_q,
        high_q
    )


    # --------------------------------------------------------
    # STANDARD SCALER
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(X)
    )


    # --------------------------------------------------------
    # GPR KERNEL
    # --------------------------------------------------------

    kernel = (

        ConstantKernel(
            1.0,
            (
                0.1,
                10.0
            )
        )

        *

        RBF(
            length_scale=1.0,
            length_scale_bounds=(
                0.1,
                10.0
            )
        )

        +

        WhiteKernel(
            noise_level=0.05,
            noise_level_bounds=(
                1e-5,
                1.0
            )
        )
    )


    gpr = GaussianProcessRegressor(

        kernel=kernel,

        normalize_y=True,

        n_restarts_optimizer=0,

        random_state=42
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    gpr.fit(
        X_scaled,
        y
    )


    # --------------------------------------------------------
    # LATEST FEATURE
    # --------------------------------------------------------

    latest = (
        d[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )


    if latest.empty:

        raise RuntimeError(
            "Latest features kosong"
        )


    x_now = (
        latest[FEATURES]
        .iloc[-1]
        .values
        .reshape(1, -1)
    )


    x_now_scaled = (
        scaler.transform(
            x_now
        )
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    pred, std = gpr.predict(

        x_now_scaled,

        return_std=True
    )


    pred_return = float(
        pred[0]
    )

    uncertainty = float(
        std[0]
    )


    return {

        "prediction":
            pred_return,

        "uncertainty":
            uncertainty,

        "rsi":
            float(
                d["rsi14"]
                .iloc[-1]
            ),

        "mom6":
            float(
                d["mom6"]
                .iloc[-1]
            ),

        "mom12":
            float(
                d["mom12"]
                .iloc[-1]
            ),

        "mom24":
            float(
                d["mom24"]
                .iloc[-1]
            ),

        "vol24":
            float(
                d["vol24"]
                .iloc[-1]
            ),

        "volume_z":
            float(
                d["volume_z"]
                .iloc[-1]
            ),

        "ema12_dist":
            float(
                d["ema12_dist"]
                .iloc[-1]
            ),

        "ema24_dist":
            float(
                d["ema24_dist"]
                .iloc[-1]
            ),

        "last_price":
            float(
                d["close"]
                .iloc[-1]
            ),

        "last_time":
            d["ts"]
            .iloc[-1]
    }


# ============================================================
# GPR SCORE
# ============================================================

def calculate_score(result):

    pred = result["prediction"]

    uncertainty = result[
        "uncertainty"
    ]

    rsi = result["rsi"]

    mom6 = result["mom6"]

    mom12 = result["mom12"]

    ema12 = result[
        "ema12_dist"
    ]

    ema24 = result[
        "ema24_dist"
    ]

    volume_z = result[
        "volume_z"
    ]


    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if pred > 0:

        direction = "LONG"

    elif pred < 0:

        direction = "SHORT"

    else:

        direction = "NEUTRAL"


    # --------------------------------------------------------
    # SIGNAL STRENGTH
    # --------------------------------------------------------

    signal_strength = min(

        abs(pred) / 0.05,

        1.0
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence_ratio = (

        abs(pred)

        /

        max(
            uncertainty,
            1e-8
        )
    )


    confidence = min(

        confidence_ratio / 3.0,

        1.0
    )


    # --------------------------------------------------------
    # MOMENTUM AGREEMENT
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RSI QUALITY
    # --------------------------------------------------------

    if direction == "LONG":

        rsi_score = (

            1.0
            if 45 <= rsi <= 68

            else 0.5
            if 35 <= rsi <= 75

            else 0.2
        )


    elif direction == "SHORT":

        rsi_score = (

            1.0
            if 32 <= rsi <= 55

            else 0.5
            if 25 <= rsi <= 65

            else 0.2
        )


    else:

        rsi_score = 0


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_score = min(

        max(
            (volume_z + 1) / 3,
            0
        ),

        1
    )


    # --------------------------------------------------------
    # FINAL SCORE
    #
    # TOTAL = 100
    #
    # GPR prediction       45
    # Confidence            25
    # Momentum               20
    # RSI                     7
    # Volume                  3
    # --------------------------------------------------------

    score = (

        45
        * signal_strength

        +

        25
        * confidence

        +

        20
        * momentum_score

        +

        7
        * rsi_score

        +

        3
        * volume_score
    )


    return (

        float(
            np.clip(
                score,
                0,
                100
            )
        ),

        direction
    )


# ============================================================
# ANALYZE ONE SYMBOL
# ============================================================

def analyze_symbol(
    inst_id,
    horizon
):

    try:

        df = get_candles(
            inst_id,
            limit=180
        )


        result = calculate_gpr(
            df,
            horizon
        )


        score, direction = (
            calculate_score(
                result
            )
        )


        symbol = (
            inst_id
            .replace(
                "-USDT-SWAP",
                ""
            )
        )


        confidence = (

            abs(
                result["prediction"]
            )

            /

            max(
                result["uncertainty"],
                1e-8
            )
        )


        return {

            "Symbol":
                symbol,

            "Instrument":
                inst_id,

            "Signal":
                direction,

            "GPR Score":
                round(
                    score,
                    1
                ),

            "GPR Return":
                result[
                    "prediction"
                ] * 100,

            "Uncertainty":
                result[
                    "uncertainty"
                ] * 100,

            "Confidence":
                confidence,

            "Price":
                result[
                    "last_price"
                ],

            "RSI":
                result[
                    "rsi"
                ],

            "Mom 6H":
                result[
                    "mom6"
                ] * 100,

            "Mom 12H":
                result[
                    "mom12"
                ] * 100,

            "Mom 24H":
                result[
                    "mom24"
                ] * 100,

            "Volume Z":
                result[
                    "volume_z"
                ],

            "EMA12 Dist":
                result[
                    "ema12_dist"
                ] * 100,

            "EMA24 Dist":
                result[
                    "ema24_dist"
                ] * 100,

            "Last Candle":
                result[
                    "last_time"
                ],

            "Status":
                "OK"
        }


    except Exception as e:

        return {

            "Symbol":
                inst_id.replace(
                    "-USDT-SWAP",
                    ""
                ),

            "Instrument":
                inst_id,

            "Signal":
                "-",

            "GPR Score":
                np.nan,

            "GPR Return":
                np.nan,

            "Uncertainty":
                np.nan,

            "Confidence":
                np.nan,

            "Price":
                np.nan,

            "RSI":
                np.nan,

            "Mom 6H":
                np.nan,

            "Mom 12H":
                np.nan,

            "Mom 24H":
                np.nan,

            "Volume Z":
                np.nan,

            "EMA12 Dist":
                np.nan,

            "EMA24 Dist":
                np.nan,

            "Last Candle":
                pd.NaT,

            "Status":
                str(e)
        }


# ============================================================
# SCAN ALL
# ============================================================

def run_scan(
    symbols,
    horizon,
    workers
):

    results = []


    progress = st.progress(
        0
    )

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
            ):
            symbol

            for symbol in symbols
        }


        for future in as_completed(
            futures
        ):

            try:

                results.append(
                    future.result()
                )


            except Exception as e:

                symbol = futures[
                    future
                ]

                results.append({

                    "Symbol":
                        symbol,

                    "Status":
                        str(e)
                })


            completed += 1


            progress.progress(

                completed
                /
                total
            )


            status_text.write(

                f"Scanning "
                f"{completed}/{total}..."
            )


    progress.empty()

    status_text.empty()


    return pd.DataFrame(
        results
    )


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

st.sidebar.title(
    "⚙️ Scanner Settings"
)


horizon = st.sidebar.selectbox(

    "Horizon GPR",

    options=[
        3,
        6,
        10,
        12,
        18,
        24
    ],

    index=2,

    format_func=lambda x:
        f"{x} candle / {x} jam"
)


coin_count = st.sidebar.selectbox(

    "Jumlah coin",

    options=[
        20,
        30,
        50,
        75,
        79
    ],

    index=4
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

    st.session_state.scan_time = None

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(

    '<div class="main-title">'
    '📈 GPR CRYPTO FUTURES SCANNER'
    '</div>',

    unsafe_allow_html=True
)


st.markdown(

    '<div class="subtitle">'
    'Target utama: USDT Perpetual · '
    'Timeframe 1H · '
    'GPR sebagai mesin ranking utama'
    '</div>',

    unsafe_allow_html=True
)


# ============================================================
# INFO
# ============================================================

st.info(

    "Mode data: public market data, tanpa API key. "
    "Universe futures menggunakan daftar static agar "
    "scanner tidak bergantung pada endpoint "
    "instruments/exchangeInfo yang sebelumnya "
    "menghasilkan HTTP 403/451."
)


# ============================================================
# SCAN
# ============================================================

if scan_button:

    universe = get_universe()

    selected_symbols = (
        universe[:coin_count]
    )


    start_time = time.time()


    df_result = run_scan(

        selected_symbols,

        horizon,

        workers
    )


    elapsed = (
        time.time()
        - start_time
    )


    st.session_state.scan_result = (
        df_result
    )

    st.session_state.scan_time = (
        elapsed
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

if st.session_state.scan_result is None:

    st.warning(
        "Belum ada hasil. "
        "Tekan **SCAN FUTURES 1H**."
    )


else:

    df = (
        st.session_state.scan_result
        .copy()
    )


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

    ranked = (
        success_df
        .copy()
    )


    if not ranked.empty:

        ranked = (

            ranked
            .sort_values(
                "GPR Score",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


    candidates = ranked[
        ranked["GPR Score"]
        >= minimum_score
    ].copy()


    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    universe_count = len(df)

    success_count = len(
        success_df
    )

    failed_count = len(
        failed_df
    )

    candidate_count = len(
        candidates
    )


    if not ranked.empty:

        top_return = ranked.iloc[0][
            "GPR Return"
        ]

    else:

        top_return = np.nan


    # ========================================================
    # TOP METRICS
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )


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

        f"Scan selesai: "
        f"{success_count} berhasil, "
        f"{failed_count} gagal "
        f"dalam "
        f"{st.session_state.scan_time:.1f} detik."
    )


    # ========================================================
    # SECOND METRICS
    # ========================================================

    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "BERHASIL DIANALISIS",
            success_count
        )


    with c2:

        st.metric(
            f"LOLOS SCORE ≥ "
            f"{minimum_score}",
            candidate_count
        )


    with c3:

        st.metric(
            "KANDIDAT",
            candidate_count
        )


    with c4:

        if np.isfinite(
            top_return
        ):

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

    st.markdown(
        '<div class="section-title">'
        '🏆 GPR Futures Ranking'
        '</div>',
        unsafe_allow_html=True
    )


    if ranked.empty:

        st.warning(
            "Tidak ada coin yang berhasil dianalisis."
        )


    else:

        display_df = (
            ranked
            .copy()
        )


        display_df = display_df[
            display_df["GPR Score"]
            >= minimum_score
        ].copy()


        if display_df.empty:

            st.warning(

                f"Belum ada coin yang "
                f"mencapai minimum score "
                f"{minimum_score}. "
                f"Turunkan threshold "
                f"untuk melihat kandidat."
            )


        else:

            display_df.insert(

                0,

                "Rank",

                range(
                    1,
                    len(display_df) + 1
                )
            )


            # ------------------------------------------------
            # DISPLAY FORMATTING
            # ------------------------------------------------

            display_df[
                "GPR Return"
            ] = display_df[
                "GPR Return"
            ].map(
                lambda x:
                    f"{x:+.2f}%"
            )


            display_df[
                "Uncertainty"
            ] = display_df[
                "Uncertainty"
            ].map(
                lambda x:
                    f"{x:.2f}%"
            )


            display_df[
                "Confidence"
            ] = display_df[
                "Confidence"
            ].map(
                lambda x:
                    f"{x:.2f}"
            )


            display_df[
                "Price"
            ] = display_df[
                "Price"
            ].map(
                lambda x:
                    f"{x:,.8g}"
            )


            display_df[
                "RSI"
            ] = display_df[
                "RSI"
            ].map(
                lambda x:
                    f"{x:.1f}"
            )


            display_df[
                "Mom 6H"
            ] = display_df[
                "Mom 6H"
            ].map(
                lambda x:
                    f"{x:+.2f}%"
            )


            display_df[
                "Mom 12H"
            ] = display_df[
                "Mom 12H"
            ].map(
                lambda x:
                    f"{x:+.2f}%"
            )


            display_df[
                "Mom 24H"
            ] = display_df[
                "Mom 24H"
            ].map(
                lambda x:
                    f"{x:+.2f}%"
            )


            display_df[
                "Volume Z"
            ] = display_df[
                "Volume Z"
            ].map(
                lambda x:
                    f"{x:+.2f}"
            )


            # ------------------------------------------------
            # RANKING COLUMNS
            # ------------------------------------------------

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

                display_df[
                    ranking_columns
                ],

                use_container_width=True,

                hide_index=True,

                column_config={

                    "Rank":
                        st.column_config.NumberColumn(
                            "🏆 Rank",
                            width="small"
                        ),

                    "Symbol":
                        st.column_config.TextColumn(
                            "COIN"
                        ),

                    "Signal":
                        st.column_config.TextColumn(
                            "SIGNAL"
                        ),

                    "GPR Score":
                        st.column_config.NumberColumn(
                            "GPR SCORE",
                            format="%.1f"
                        ),

                    "GPR Return":
                        st.column_config.TextColumn(
                            "GPR RETURN"
                        ),

                    "Uncertainty":
                        st.column_config.TextColumn(
                            "UNCERTAINTY"
                        ),

                    "Confidence":
                        st.column_config.TextColumn(
                            "CONFIDENCE"
                        ),

                    "Price":
                        st.column_config.TextColumn(
                            "PRICE"
                        ),

                    "RSI":
                        st.column_config.TextColumn(
                            "RSI"
                        ),

                    "Mom 6H":
                        st.column_config.TextColumn(
                            "MOM 6H"
                        ),

                    "Mom 12H":
                        st.column_config.TextColumn(
                            "MOM 12H"
                        ),

                    "Mom 24H":
                        st.column_config.TextColumn(
                            "MOM 24H"
                        ),

                    "Volume Z":
                        st.column_config.TextColumn(
                            "VOL Z"
                        ),
                }
            )


    # ========================================================
    # TOP GPR SIGNALS
    #
    # PENTING:
    # Tidak menggunakan HTML <div>.
    # Menggunakan native Streamlit sehingga tidak
    # akan muncul sebagai source code lagi.
    # ========================================================

    if not candidates.empty:

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🎯 Top GPR Signals'
            '</div>',
            unsafe_allow_html=True
        )


        top_n = min(
            5,
            len(candidates)
        )


        top = (
            candidates
            .head(top_n)
        )


        cols = st.columns(
            top_n
        )


        for col, (_, row) in zip(
            cols,
            top.iterrows()
        ):

            with col:

                signal = row[
                    "Signal"
                ]


                if signal == "LONG":

                    signal_text = (
                        "🟢 LONG"
                    )

                elif signal == "SHORT":

                    signal_text = (
                        "🔴 SHORT"
                    )

                else:

                    signal_text = (
                        "🟡 NEUTRAL"
                    )


                # Native Streamlit card
                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### {row['Symbol']}"
                    )


                    st.metric(

                        "GPR SCORE",

                        f"{row['GPR Score']:.1f}"
                    )


                    st.write(
                        f"**Signal:** "
                        f"{signal_text}"
                    )


                    st.write(

                        f"**GPR Return:** "
                        f"{row['GPR Return']:+.2f}%"
                    )


                    st.write(

                        f"**Uncertainty:** "
                        f"{row['Uncertainty']:.2f}%"
                    )


                    st.write(

                        f"**Confidence:** "
                        f"{row['Confidence']:.2f}"
                    )


    # ========================================================
    # DETAIL ANALYSIS
    # ========================================================

    st.markdown("---")


    st.markdown(
        '<div class="section-title">'
        '🔎 Detail Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    if not ranked.empty:

        selected_symbol = (
            st.selectbox(
                "Pilih coin",
                ranked[
                    "Symbol"
                ].tolist()
            )
        )


        detail = ranked[
            ranked["Symbol"]
            == selected_symbol
        ].iloc[0]


        d1, d2, d3, d4 = (
            st.columns(4)
        )


        with d1:

            st.metric(
                "SIGNAL",
                detail["Signal"]
            )


        with d2:

            st.metric(
                "GPR SCORE",
                f'{detail["GPR Score"]:.1f}'
            )


        with d3:

            st.metric(
                "FORECAST",
                f'{detail["GPR Return"]:+.2f}%'
            )


        with d4:

            st.metric(
                "UNCERTAINTY",
                f'{detail["Uncertainty"]:.2f}%'
            )


        st.markdown("")


        d1, d2, d3, d4 = (
            st.columns(4)
        )


        with d1:

            st.write(
                f"**Harga**  \n"
                f"{detail['Price']:,.8g}"
            )


        with d2:

            st.write(
                f"**RSI 14**  \n"
                f"{detail['RSI']:.1f}"
            )


        with d3:

            st.write(
                f"**Momentum 6H**  \n"
                f"{detail['Mom 6H']:+.2f}%"
            )


        with d4:

            st.write(
                f"**Momentum 12H**  \n"
                f"{detail['Mom 12H']:+.2f}%"
            )


        d1, d2, d3, d4 = (
            st.columns(4)
        )


        with d1:

            st.write(
                f"**Momentum 24H**  \n"
                f"{detail['Mom 24H']:+.2f}%"
            )


        with d2:

            st.write(
                f"**Volume Z**  \n"
                f"{detail['Volume Z']:+.2f}"
            )


        with d3:

            st.write(
                f"**EMA12 Distance**  \n"
                f"{detail['EMA12 Dist']:+.2f}%"
            )


        with d4:

            st.write(
                f"**EMA24 Distance**  \n"
                f"{detail['EMA24 Dist']:+.2f}%"
            )


        st.caption(
            f"Last confirmed candle: "
            f"{detail['Last Candle']}"
        )


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    st.markdown("---")


    with st.expander(
        "📋 ERROR / DIAGNOSTIC"
    ):

        if failed_df.empty:

            st.success(
                "Tidak ada error "
                "pada coin yang discan."
            )


        else:

            error_display = (
                failed_df[
                    [
                        "Symbol",
                        "Status"
                    ]
                ]
                .copy()
            )


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
    "USDT Perpetual · 1H · "
    "Public Market Data · "
    "Research Prototype"
)
