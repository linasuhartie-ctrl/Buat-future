import math
import numpy as np
import pandas as pd


def ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def matern32_kernel(x, z, amplitude, scale):
    d = x - z
    distance2 = d[0]**2 + 0.8*d[1]**2 + 0.6*d[2]**2 + 0.8*d[3]**2 + 0.6*d[4]**2
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
                prior = values[max(0, i-length+1):i+1]
                if np.isfinite(prior).sum() == length:
                    out[i] = float(np.mean(prior))
            else:
                out[i] = alpha * values[i] + (1.0-alpha) * prev
    return out


def gpr_profile(timeframe):
    if timeframe in ("5m", "15m"):
        return dict(training=30, momentum_len=6, vol_len=24, ema_fast=8, ema_slow=26,
                    kernel_scale=1.8, noise_ratio=0.85, volume_fade=0.35)
    if timeframe == "1h":
        return dict(training=35, momentum_len=5, vol_len=20, ema_fast=8, ema_slow=21,
                    kernel_scale=1.6, noise_ratio=0.65, volume_fade=0.50)
    return dict(training=30, momentum_len=5, vol_len=14, ema_fast=8, ema_slow=21,
                kernel_scale=1.5, noise_ratio=0.30, volume_fade=0.65)


def project(df, horizon=10, validation_candles=3, training=30, momentum_len=5,
            vol_len=14, ema_fast=8, ema_slow=21, kernel_scale=1.5,
            noise_ratio=0.30, volume_fade=0.65, band_sigma=1.96,
            atr_width=0.70, band_mode="Pita GPR"):
    min_needed = max(training + vol_len + ema_slow + 10, 60)
    idx = len(df) - 1 - validation_candles
    if idx < min_needed:
        raise ValueError(f"Data candle hanya {len(df)}; perlu minimal {min_needed + validation_candles}.")

    close = df["close"].astype(float).to_numpy()
    volume = df["volume"].fillna(0).astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()

    logret = np.full(len(close), np.nan)
    logret[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12))
    ret_series = pd.Series(logret)
    momentum = ret_series.ewm(span=momentum_len, adjust=False, min_periods=1).mean().to_numpy()
    volatility = ret_series.rolling(vol_len, min_periods=vol_len).std(ddof=0).to_numpy()
    vol_base = pd.Series(volatility).rolling(training, min_periods=training).mean().to_numpy()
    avg_volume = pd.Series(volume).rolling(vol_len, min_periods=vol_len).mean().to_numpy()
    rel_volume = np.zeros(len(close))
    valid = (volume > 0) & (avg_volume > 0) & np.isfinite(avg_volume)
    rel_volume[valid] = np.log(volume[valid] / avg_volume[valid])
    fast = ema(pd.Series(close), ema_fast).to_numpy()
    slow = ema(pd.Series(close), ema_slow).to_numpy()
    direction = (fast - slow) / np.maximum(close, 1e-12)

    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr[0] = high[0] - low[0]
    atr = pine_rma(tr, vol_len)

    vol_ref = max(float(vol_base[idx]) if np.isfinite(vol_base[idx]) else 0.0, 0.0001)
    if not np.isfinite(volatility[idx]) or not np.isfinite(atr[idx]) or not np.isfinite(logret[idx]):
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

    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    if n < 10:
        raise ValueError("Training sample GPR terlalu sedikit.")

    signal_var = max(float(np.mean(y*y)), 1e-8)
    noise_var = signal_var * noise_ratio + 1e-10
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            value = matern32_kernel(X[i], X[j], signal_var, kernel_scale)
            if i == j:
                value += noise_var
            K[i, j] = K[j, i] = value

    K += np.eye(n) * 1e-6
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        K += np.eye(n) * 1e-4
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        except np.linalg.LinAlgError as exc:
            raise ValueError("Matriks GPR tidak stabil secara numerik.") from exc

    price_start = float(close[idx])
    price_prev = price_start
    ret_prev = float(logret[idx])
    mom_prev = float(momentum[idx])
    fast_prev, slow_prev = float(fast[idx]), float(slow[idx])
    volume_prev = float(rel_volume[idx])
    vol_prev = float(volatility[idx])
    cumulative_return = 0.0
    cumulative_variance = 0.0
    predictions = []

    total_steps = horizon + validation_candles
    for h in range(total_steps):
        fade = volume_fade ** h
        assumed_vol = vol_ref + (vol_prev - vol_ref) * fade
        x_star = np.array([
            ret_prev / vol_ref,
            mom_prev / vol_ref,
            volume_prev * fade,
            assumed_vol / vol_ref - 1.0,
            ((fast_prev - slow_prev) / max(price_prev, 1e-12)) / vol_ref,
        ])
        kstar = np.array([matern32_kernel(X[i], x_star, signal_var, kernel_scale) for i in range(n)])
        pred_ret = float(kstar @ alpha)
        solved = np.linalg.solve(L, kstar)
        predictive_variance = max(float(signal_var + noise_var - solved @ solved), 1e-12)
        cumulative_variance += predictive_variance
        cumulative_return += pred_ret
        price_pred = price_start * math.exp(cumulative_return)
        sigma_cum = math.sqrt(max(cumulative_variance, 0.0))

        if band_mode == "Kanal ATR":
            distance = atr_width * float(atr[idx]) * math.sqrt(h + 1.0)
            upper = price_pred + distance
            lower = max(1e-12, price_pred - distance)
        else:
            upper = price_pred * math.exp(band_sigma * sigma_cum)
            lower = price_pred * math.exp(-band_sigma * sigma_cum)

        real_idx = idx + h + 1
        actual_price = float(close[real_idx]) if real_idx < len(df) else None
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
        mom_prev = a_m * pred_ret + (1.0-a_m) * mom_prev
        a_f = 2.0 / (ema_fast + 1.0)
        a_s = 2.0 / (ema_slow + 1.0)
        fast_prev = a_f * price_pred + (1.0-a_f) * fast_prev
        slow_prev = a_s * price_pred + (1.0-a_s) * slow_prev

    latest = float(close[-1])
    trend = "BULLISH" if fast[-1] > slow[-1] else "BEARISH"
    mom_now = float(momentum[-1]) if np.isfinite(momentum[-1]) else 0.0
    vol_now = float(volume[-1])
    vol_avg = float(avg_volume[-1]) if np.isfinite(avg_volume[-1]) else 0.0
    rel_vol = vol_now / vol_avg if vol_avg > 0 else 0.0

    validation = [p for p in predictions if not p["is_future"] and p["actual_price"] is not None]
    errors = [abs(p["price"] - p["actual_price"]) / max(p["actual_price"], 1e-12) for p in validation]
    hits = sum(p["lower"] <= p["actual_price"] <= p["upper"] for p in validation)
    mape = float(np.mean(errors)) if errors else 1.0
    hit_rate = hits / len(validation) if validation else 0.0
    final = predictions[-1]
    cumulative_sigma = math.sqrt(max(cumulative_variance, 0.0))
    uncertainty = 100.0 * cumulative_sigma / (abs(cumulative_return) + cumulative_sigma + 1e-10)

    return {
        "price": latest,
        "projection_start_price": price_start,
        "trend": trend,
        "momentum": "POSITIVE" if mom_now > 0 else "NEGATIVE",
        "relative_volume": rel_vol,
        "gpr_return_pct": (final["price"] / latest - 1.0) * 100.0,
        "uncertainty_pct": uncertainty,
        "validation_mape": mape,
        "validation_hit_rate": hit_rate,
        "projected_price": final["price"],
        "horizon": horizon,
        "predictions": predictions,
    }
