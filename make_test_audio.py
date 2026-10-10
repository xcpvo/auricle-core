"""Генерирует синтетические WAV для smoke-теста."""
import numpy as np
import soundfile as sf
from scipy.signal import chirp

SR = 44100


def _save(path, y):
    y = y / np.max(np.abs(y)) * 0.9
    sf.write(path, y.astype(np.float32), SR)
    print(f'  {path}')


def _chirp(duration, f0, f1):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    return chirp(t, f0=f0, f1=f1, t1=duration, method='logarithmic')


def _chord(duration, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    y = sum(rng.random() * np.sin(2 * np.pi * f * t + rng.random())
            for f in (220, 330, 440, 660, 880, 1320))
    y += 0.05 * rng.standard_normal(y.shape)
    return y


def main():
    print('Треки:')
    a = _chirp(30, 200, 4000)
    b = _chord(30, seed=42)
    c = _chirp(30, 4000, 200)
    _save('track_a.wav', a)
    _save('track_b.wav', b)
    _save('track_c.wav', c)

    print('Запросы:')
    _save('query_a.wav', a[SR * 12 : SR * 22])
    _save('query_b.wav', b[SR * 5  : SR * 15])
    _save('query_c.wav', c[SR * 8  : SR * 18])


if __name__ == '__main__':
    main()