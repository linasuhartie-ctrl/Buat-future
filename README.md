# GPR Market Lab

Prototype bersih yang menjadikan **Gaussian Process Regression (GPR)** sebagai inti model.

## Struktur

- `app.py` — UI Streamlit
- `gpr_model.py` — kernel Matérn 3/2, feature engineering, training GPR, projection, validation
- `data.py` — pengambilan dan normalisasi data Yahoo Finance
- `requirements.txt` — dependency

## Jalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Konsep model

Input feature GPR:
1. log return
2. momentum EMA
3. relative volume
4. volatility relatif
5. jarak EMA fast/slow

Target model: **log return candle berikutnya**.

Kernel: **Matérn 3/2**.

Output:
- proyeksi harga
- rentang ketidakpastian
- validasi historis T-n
- MAPE validasi
- hit-rate band

Versi ini sengaja tidak memasukkan universe scanner besar, multi-threading, atau API exchange sebagai fondasi. Setelah core GPR stabil, scanner dapat ditambahkan sebagai layer di atasnya.
