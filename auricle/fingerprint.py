"""Fingerprinter: аудио → (hash, offset).

Изменения v2:
- Логарифмические частотные бины — устойчивее к эквалайзеру и микрофону.
- Spectral whitening — вычитаем сглаженный спектр, чтобы убрать АЧХ тракта.
- 32-битный хеш: log_f1(8) | log_f2(8) | dt(8) | mag1(4) | mag2(4).
"""
from typing import List, Tuple
import numpy as np
from scipy.signal import stft
from scipy.ndimage import maximum_filter, uniform_filter1d

from .audio import load_audio
from .config import Config

Hash = Tuple[int, int]  # (hash, offset_в_кадрах)


class Fingerprinter:
    def __init__(self, config: Config):
        self.config = config
        self._bin_map = self._build_log_bin_map()

    # --- публичное API ---

    def from_file(self, path: str) -> List[Hash]:
        y, _ = load_audio(path, self.config.target_sr)
        return self.from_samples(y)

    def from_samples(self, y: np.ndarray) -> List[Hash]:
        S_raw = self._spectrogram(y)
        if self.config.whiten:
            S = self._whiten(S_raw)
            threshold = self.config.whiten_threshold_db
        else:
            S = S_raw
            threshold = self.config.peak_threshold_db
        times, freqs = self._find_peaks(S, threshold)
        mags = self._quantize_magnitudes(S, times, freqs)
        return self._make_hashes(times, freqs, mags)

    # --- отладочное ---

    def spectrogram(self, y: np.ndarray):
        return self._spectrogram(y)

    def peaks(self, y: np.ndarray):
        S_raw = self._spectrogram(y)
        S = self._whiten(S_raw) if self.config.whiten else S_raw
        threshold = self.config.whiten_threshold_db if self.config.whiten \
                    else self.config.peak_threshold_db
        return self._find_peaks(S, threshold)

    # --- внутреннее ---

    def _build_log_bin_map(self):
        """FFT-бин → лог-бин (0..log_n_bins-1). -1 = вне полосы."""
        cfg = self.config
        n_freqs = cfg.n_fft // 2 + 1
        freqs = np.arange(n_freqs) * cfg.target_sr / cfg.n_fft

        bin_map = np.full(n_freqs, -1, dtype=np.int32)
        mask = (freqs >= cfg.log_f_min) & (freqs <= cfg.log_f_max)
        pos = np.log(freqs[mask] / cfg.log_f_min) / np.log(cfg.log_f_max / cfg.log_f_min)
        log_bins = np.clip((pos * cfg.log_n_bins).astype(np.int32),
                           0, cfg.log_n_bins - 1)
        bin_map[mask] = log_bins
        return bin_map

    def _spectrogram(self, y: np.ndarray) -> np.ndarray:
        cfg = self.config
        _, _, Z = stft(
            y, fs=cfg.target_sr,
            nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop,
            window=cfg.window,
            boundary=None, padded=False,
        )
        return 20.0 * np.log10(np.abs(Z) + 1e-10)

    def _whiten(self, S: np.ndarray) -> np.ndarray:
        """Вычитаем сглаженный по частоте спектр.

        Пики становятся локальными отклонениями от базовой линии —
        независимо от общей АЧХ (колонка, микрофон, эквалайзер).
        """
        w = self.config.whiten_window
        smooth = uniform_filter1d(S, size=w, axis=0, mode='nearest')
        return S - smooth

    def _find_peaks(self, S: np.ndarray, threshold: float):
        cfg = self.config
        local_max = maximum_filter(S, size=(cfg.peak_neighborhood,
                                             cfg.peak_neighborhood))
        peaks = (S == local_max) & (S > threshold)
        m = cfg.peak_neighborhood // 2
        peaks[:m, :] = False
        peaks[-m:, :] = False
        peaks[:, :m] = False
        peaks[:, -m:] = False
        freqs, times = np.where(peaks)
        order = np.argsort(times, kind='stable')
        return times[order].astype(np.int32), freqs[order].astype(np.int32)

    def _quantize_magnitudes(self, S, times, freqs) -> np.ndarray:
        """Квантуем магнитуду в N бакетов (0..N-1). Используем
        квантили по всему кадру, чтобы бакеты были устойчивы к громкости."""
        vals = S[freqs, times]
        if len(vals) == 0:
            return np.zeros(0, dtype=np.int32)
        # Нормируем по перцентилям, чтобы не зависеть от абсолютной шкалы
        lo = np.percentile(vals, 5)
        hi = np.percentile(vals, 95)
        if hi - lo < 1e-6:
            return np.zeros(len(vals), dtype=np.int32)
        norm = (vals - lo) / (hi - lo)
        buckets = np.clip((norm * self.config.mag_n_buckets).astype(np.int32),
                          0, self.config.mag_n_buckets - 1)
        return buckets

    def _make_hashes(self, times, freqs, mags) -> List[Hash]:
        cfg = self.config
        bin_map = self._bin_map
        log_freqs = bin_map[freqs]  # -1 если вне полосы
        hashes: List[Hash] = []
        n = len(times)

        for i in range(n):
            t1 = int(times[i])
            f1 = int(log_freqs[i])
            if f1 < 0:
                continue
            m1 = int(mags[i]) & 0xF
            targets = 0
            for j in range(i + 1, n):
                t2 = int(times[j])
                dt = t2 - t1
                if dt < cfg.dt_min:
                    continue
                if dt > cfg.dt_max:
                    break
                f2 = int(log_freqs[j])
                if f2 < 0:
                    continue
                m2 = int(mags[j]) & 0xF
                # 32-битный хеш: 8+8+8+4+4
                h = (f1 << 24) | (f2 << 16) | (dt << 8) | (m1 << 4) | m2
                hashes.append((h, t1))
                targets += 1
                if targets >= cfg.max_targets:
                    break
        return hashes