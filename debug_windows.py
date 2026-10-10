"""Диагностика live: печатает матчи для каждого окна simulate."""
import sys
from pathlib import Path

import numpy as np

from auricle import Config, Engine
from auricle.audio import load_audio

PATH = sys.argv[1] if len(sys.argv) > 1 else 'wav/Зависим.wav'
WINDOW_SEC = 10
HOP_SEC = 2

cfg = Config()
cfg.top_k = 30   # больше, чтобы видеть что там есть

with Engine('fingerprints.db', cfg) as eng:
    sr = cfg.target_sr
    y, _ = load_audio(PATH, sr)
    print(f'loaded {len(y)/sr:.1f}s @ {sr} Hz')

    hop = HOP_SEC * sr
    win = WINDOW_SEC * sr

    for i, start in enumerate(range(0, len(y) - win + 1, hop)):
        window = y[start:start + win].copy()
        peak = float(np.max(np.abs(window)))
        if peak > 1e-6:
            window = window / peak * 0.9

        matches = eng.search_samples(window)
        t = start / sr
        print(f'\n[{i:3}] t={t:6.1f}s  matches={len(matches)}')
        for m in matches[:5]:
            delta_s = m.delta * cfg.frame_duration
            print(f'      {m.title[:24]:24}  score={m.score:5}  '
                  f'delta={delta_s:+7.1f}s  cov={m.coverage:.0%}')