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
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1 { font-size: 2.35rem !important; margin-bottom: .15rem !important; }
    .muted { color: #777; font-size: .9rem; }
    .long { color: #0a9f5a; font-weight: 800; }
    .short { color: #e53935; font-weight: 800; }
    .guide { padding: 14px 18px; border: 1px solid #ddd; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

TIMEFRAME_CONFIG = {
    "15m": {"bar": "15m", "minutes": 15},
    "1H": {"bar": "1H", "minutes": 60},
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
# HELPERS
# ============================================================
def timeframe_bars_for_hours(hours, timeframe):
    minutes = TIMEFRAME_CONFIG[timeframe]["minutes"]
    return max(1, int(round(hours * 60 / minutes)))


def format_horizon(horizon_bars, timeframe):
    hours = horizon_bars * TIMEFRAME_CONFIG[timeframe]["minutes"] / 60
    return f"{horizon_bars} candle / {hours:g} jam"


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def calculate_confidence(prediction, uncertainty):
    uncertainty = max(abs(float(uncertainty)), 1e-9)
    prediction = float(prediction)
    ratio = abs(prediction) / uncertainty

    # Ini probabilitas sisi forecast menurut asumsi Gaussian.
    # BUKAN probabilitas profit.
    probability = float(np.clip(normal_cdf(ratio), 0.50, 0.9999))
