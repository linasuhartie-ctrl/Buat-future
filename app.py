import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data import load_history
from gpr_model import gpr_profile, project

st.set_page_config(page_title="GPR Market Lab", page_icon="📈", layout="wide")

st.title("📈 GPR MARKET LAB")
st.caption("Prototype dasar berbasis Gaussian Process Regression — fokus ke metode GPR dulu, scanner multi-market belakangan.")

with st.sidebar:
    st.header("Input model")
    symbol = st.text_input("Ticker Yahoo Finance", "BBCA.JK")
    timeframe_label = st.selectbox("Timeframe", ["Harian", "1 jam", "15 menit", "5 menit"])
    timeframe = {"Harian":"1d", "1 jam":"1h", "15 menit":"15m", "5 menit":"5m"}[timeframe_label]
    horizon = st.slider("Horizon candle", 1, 20, 10)
    validation = st.slider("Validasi T-", 1, 10, 3)
    band_mode = st.selectbox("Mode band", ["Pita GPR", "Kanal ATR"])
    run = st.button("🚀 Jalankan GPR", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Mengambil data dan menghitung GPR..."):
            hist = load_history(symbol.strip().upper(), timeframe)
            if hist.empty:
                raise ValueError("Data historis kosong atau ticker tidak ditemukan.")
            result = project(hist, horizon=horizon, validation_candles=validation,
                             band_mode=band_mode, **gpr_profile(timeframe))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Harga", f"{result['price']:,.4f}")
        c2.metric("GPR Return", f"{result['gpr_return_pct']:+.2f}%")
        c3.metric("Validasi hit-rate", f"{result['validation_hit_rate']*100:.1f}%")
        c4.metric("Uncertainty", f"{result['uncertainty_pct']:.1f}%")

        st.subheader(f"{symbol.upper()} — {timeframe_label}")
        preds = result["predictions"]
        rows = []
        valid_count = sum(not p["is_future"] for p in preds)
        for p in preds:
            if p["is_future"]:
                label = f"Future +{p['step']-valid_count}"
            else:
                distance = valid_count - p["step"]
                label = f"T-{distance}" if distance > 0 else "T0"
            rows.append({"Candle": label, "GPR": p["price"], "Actual": p["actual_price"],
                         "Upper": p["upper"], "Lower": p["lower"]})

        chart = go.Figure()
        chart.add_trace(go.Scatter(x=[r["Candle"] for r in rows], y=[r["GPR"] for r in rows], name="GPR"))
        chart.add_trace(go.Scatter(x=[r["Candle"] for r in rows], y=[r["Actual"] for r in rows], name="Actual", connectgaps=False))
        chart.add_trace(go.Scatter(x=[r["Candle"] for r in rows], y=[r["Upper"] for r in rows], name="Upper", line=dict(dash="dot")))
        chart.add_trace(go.Scatter(x=[r["Candle"] for r in rows], y=[r["Lower"] for r in rows], name="Lower", line=dict(dash="dot")))
        chart.update_layout(height=520, margin=dict(l=10,r=10,t=20,b=10), hovermode="x unified")
        st.plotly_chart(chart, use_container_width=True)

        st.subheader("Output GPR")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("Parameter model"):
            st.json(gpr_profile(timeframe))
    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Masukkan ticker lalu tekan **Jalankan GPR**. Contoh: BBCA.JK, AAPL, BTC-USD.")

st.caption("Catatan: GPR di sini adalah model eksperimen untuk proyeksi return, bukan probabilitas profit dan bukan rekomendasi investasi.")
