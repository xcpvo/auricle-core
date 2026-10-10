"""Загрузка аудио: моно + ресемплинг + нормализация."""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load_audio(path, target_sr=8000):
    """Читает WAV/FLAC, усредняет стерео в моно, ресемплит в target_sr.

    Обходит проблему libsndfile с не-ASCII путями на Windows:
    открываем файл через Python (pathlib), а soundfile'у отдаём file object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Файл не найден: {path}')

    # Ключевой момент: открываем файл через Python, отдаём file object.
    # libsndfile тогда не трогает путь напрямую и не спотыкается о кириллицу.
    with open(path, 'rb') as f:
        y, sr = sf.read(f, dtype='float32', always_2d=False)

    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size == 0:
        raise ValueError(f'Пустой файл: {path}')
    if sr != target_sr:
        g = gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)

    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = y / peak * 0.9

    return y, target_sr