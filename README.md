# GPR Crypto Futures 1H Scanner

Screener Streamlit khusus **USDT Perpetual Crypto Futures** dengan **GPR sebagai mesin utama ranking**.

## Fokus v1

- Binance Futures atau Bybit Linear Futures
- Timeframe tetap **1H**
- Hanya kontrak **USDT Perpetual**
- Filter likuiditas dilakukan sebelum GPR
- GPR memakai basis kode yang kamu kirim:
  - Matern 3/2 kernel
  - momentum
  - volatility
  - relative volume
  - EMA fast/slow
  - validation T-3
  - prediction band
- Ranking:
  - validation hit rate
  - validation MAPE
  - trend
  - momentum
  - projected GPR return
- Signal:
  - LONG CANDIDATE
  - SHORT CANDIDATE
  - WATCH
  - AVOID
- Tidak memakai API key.
- Tidak melakukan order otomatis.

## Deploy ke Streamlit Cloud

1. Buat repo GitHub baru.
2. Upload:
   - `app.py`
   - `requirements.txt`
3. Di Streamlit Community Cloud pilih repo tersebut.
4. Main file: `app.py`.
5. Deploy.

## Catatan

Untuk versi awal, GPR sengaja tidak dicampur dengan terlalu banyak indikator. Data futures hanya dipakai untuk membangun universe dan filter likuiditas. Setelah versi dasar stabil, kita bisa menambahkan funding rate, open interest, taker flow, dan market regime sebagai layer terpisah tanpa merusak core GPR.
