"""
denoise_csv.py
Применяет разные методы подавления шума к числовым столбцам CSV и сохраняет результаты.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Необязательные библиотеки (установить, если нужно):
# pip install scipy pywt
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
import scipy.stats as stats
try:
    import pywt
    _HAVE_PYWT = True
except Exception:
    _HAVE_PYWT = False

def rolling_smooth(s, window=5):
    return s.rolling(window=window, center=True, min_periods=1).mean()

def savgol_smooth(s, window_length=11, polyorder=3):
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl < 3:
        wl = 3
    return pd.Series(savgol_filter(s.fillna(method='ffill').fillna(method='bfill'), wl, polyorder),
                     index=s.index)

def median_smooth(s, kernel_size=5):
    ks = int(kernel_size)
    if ks % 2 == 0:
        ks += 1
    return pd.Series(medfilt(s.fillna(method='ffill').fillna(method='bfill'), kernel_size=ks),
                     index=s.index)

def butter_lowpass_filter(s, cutoff, fs=1.0, order=4):
    # cutoff: частота среза (в тех же единицах, что и fs); если индекс равномерный, fs=1/dt
    nyq = 0.5 * fs
    normal_cutoff = float(cutoff) / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, s.fillna(method='ffill').fillna(method='bfill'))
    return pd.Series(y, index=s.index)

def remove_outliers_iqr(s, factor=1.5):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    s_clean = s.copy()
    mask = (s < low) | (s > high)
    s_clean[mask] = np.nan
    return s_clean.interpolate().fillna(method='bfill').fillna(method='ffill')

def remove_outliers_zscore(s, thr=3.0):
    z = np.abs(stats.zscore(s.fillna(method='ffill').fillna(method='bfill')))
    mask = z > thr
    s_clean = s.copy()
    s_clean[mask] = np.nan
    return s_clean.interpolate().fillna(method='bfill').fillna(method='ffill')

def wavelet_denoise(s, wavelet='db4', level=None):
    if not _HAVE_PYWT:
        raise RuntimeError("pywt не установлен (pip install pywavelets)")
    data = s.fillna(method='ffill').fillna(method='bfill').values
    coeffs = pywt.wavedec(data, wavelet, mode='per')
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    rec = pywt.waverec(denoised_coeffs, wavelet, mode='per')
    return pd.Series(rec[:len(data)], index=s.index)

def denoise_series(s, method='savgol', **kwargs):
    if method == 'rolling':
        return rolling_smooth(s, window=kwargs.get('window', 5))
    if method == 'median':
        return median_smooth(s, kernel_size=kwargs.get('kernel_size', 5))
    if method == 'savgol':
        return savgol_smooth(s, window_length=kwargs.get('window_length', 11), polyorder=kwargs.get('polyorder', 3))
    if method == 'butter':
        return butter_lowpass_filter(s, cutoff=kwargs.get('cutoff', 0.1), fs=kwargs.get('fs', 1.0), order=kwargs.get('order', 4))
    if method == 'iqr':
        return remove_outliers_iqr(s, factor=kwargs.get('factor', 1.5))
    if method == 'zscore':
        return remove_outliers_zscore(s, thr=kwargs.get('thr', 3.0))
    if method == 'wavelet':
        return wavelet_denoise(s, wavelet=kwargs.get('wavelet', 'db4'))
    raise ValueError("Unknown method")

def denoise_csv(input_csv, output_csv, method='savgol', columns=None, plot=False, plot_cols=None, **kwargs):
    df = pd.read_csv(input_csv)
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        cols = [c for c in columns if c in df.columns]
    else:
        cols = numcols
    df_clean = df.copy()
    for c in cols:
        try:
            df_clean[c] = denoise_series(df[c], method=method, **kwargs)
        except Exception as e:
            print(f"Warning: не удалось обработать столбец {c}: {e}")
    df_clean.to_csv(output_csv, index=False)
    if plot:
        if plot_cols is None:
            plot_cols = cols[:3]
        n = len(plot_cols)
        fig, axs = plt.subplots(n, 1, figsize=(8, 3*n))
        if n == 1:
            axs = [axs]
        for ax, c in zip(axs, plot_cols):
            ax.plot(df[c], label='orig', alpha=0.6)
            ax.plot(df_clean[c], label='clean', linewidth=1.5)
            ax.set_title(c)
            ax.legend()
        plt.tight_layout()
        plt.show()
    return df_clean

# Пример использования:
denoise_csv('/home/l3v/repo/Pelevin-Podmoskovnov-Lukin/blood/30-50pridurok.csv', '30-50pridurok_clean.csv', method='butter', kernel_size=7, plot=True)
