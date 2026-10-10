"""Визуализация работы ядра auricle-core.

Команды:
    python visualize.py waveform <file>
    python visualize.py spectrogram <file>
    python visualize.py whiten <file>
    python visualize.py delta <query>
    python visualize.py hashes <file>
    python visualize.py all <file>

По умолчанию графики показываются окном. С --save сохраняются в PNG.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    print('matplotlib не установлен. Установи: pip install matplotlib',
          file=sys.stderr)
    sys.exit(1)

from auricle import Config, Engine
from auricle.audio import load_audio
from auricle.fingerprint import Fingerprinter


# ─── общие помощники ────────────────────────────────────────────────────

def _setup_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#111111',
        'axes.facecolor': '#0a0a0a',
        'axes.edgecolor': '#444',
        'grid.color': '#2a2a2a',
        'font.size': 10,
    })


def _maybe_show(save: Optional[str]):
    """Сохранить или показать. Если save задан — сохраняем и не блокируемся."""
    if save:
        plt.savefig(save, dpi=130, bbox_inches='tight',
                    facecolor='#111111')
        print(f'  сохранено: {save}')
    else:
        plt.show()
    plt.close('all')


def _load(file, cfg):
    y, sr = load_audio(file, cfg.target_sr)
    return y, sr


def _format_time(ax, x, sr):
    """Подписать ось X в секундах."""
    n = len(x)
    ax.set_xlim(0, n / sr)


# ─── 1. waveform ────────────────────────────────────────────────────────

def plot_waveform(file: str, save: Optional[str] = None):
    cfg = Config()
    y, sr = _load(file, cfg)
    t = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(t, y, color='#00d4aa', linewidth=0.4)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Амплитуда')
    ax.set_title(f'Форма волны: {Path(file).name}  '
                 f'({len(y) / sr:.1f} с, {sr} Гц)')
    ax.grid(alpha=0.2)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    _maybe_show(save)


# ─── 2. spectrogram + peaks ─────────────────────────────────────────────

def plot_spectrogram(file: str, save: Optional[str] = None):
    cfg = Config()
    fp = Fingerprinter(cfg)
    y, sr = _load(file, cfg)

    S_raw = fp.spectrogram(y)
    S_whiten = fp._whiten(S_raw) if cfg.whiten else S_raw

    times, freqs = fp.peaks(y)

    # Оси
    duration = len(y) / sr
    nyq = sr / 2
    extent = [0, duration, 0, nyq]

    peak_t = times * cfg.hop / cfg.target_sr
    peak_f = freqs * cfg.target_sr / cfg.n_fft

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Верх: сырая спектрограмма
    ax = axes[0]
    im = ax.imshow(S_raw, aspect='auto', origin='lower', extent=extent,
                   cmap='magma', vmin=-80, vmax=0)
    ax.set_title(f'Сырая спектрограмма (STFT окно={cfg.n_fft}, шаг={cfg.hop})')
    ax.set_ylabel('Частота (Гц)')
    ax.set_xlim(0, duration)
    fig.colorbar(im, ax=ax, label='dB', fraction=0.025)

    # Низ: whitened + peaks
    ax = axes[1]
    v = 12
    im = ax.imshow(S_whiten, aspect='auto', origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-v, vmax=v)
    ax.scatter(peak_t, peak_f, s=3, c='lime', alpha=0.7,
               label=f'пики: {len(times)}')
    ax.set_title(f'Whitened спектрограмма + пики '
                 f'(порог {cfg.whiten_threshold_db} dB над базовой линией)')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Частота (Гц)')
    ax.set_xlim(0, duration)
    ax.set_ylim(0, nyq)
    ax.legend(loc='upper right', framealpha=0.3)
    fig.colorbar(im, ax=ax, label='дБ над локальной базой', fraction=0.025)

    fig.tight_layout()
    _maybe_show(save)


# ─── 3. whiten: до / после ──────────────────────────────────────────────

def plot_whiten(file: str, save: Optional[str] = None):
    cfg = Config()
    fp = Fingerprinter(cfg)
    y, sr = _load(file, cfg)

    S_raw = fp.spectrogram(y)
    S_whiten = fp._whiten(S_raw)

    duration = len(y) / sr
    nyq = sr / 2
    extent = [0, duration, 0, nyq]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={'height_ratios': [1, 1, 0.8]})

    # Сырая
    ax = axes[0]
    im = ax.imshow(S_raw, aspect='auto', origin='lower', extent=extent,
                   cmap='magma', vmin=-80, vmax=0)
    ax.set_title('1. Сырая спектрограмма — видны громкие и тихие полосы')
    ax.set_ylabel('Частота (Гц)')
    fig.colorbar(im, ax=ax, label='dB', fraction=0.025)

    # Сглаженная (то, что вычитается)
    w = cfg.whiten_window
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(S_raw, size=w, axis=0, mode='nearest')

    ax = axes[1]
    im = ax.imshow(smooth, aspect='auto', origin='lower', extent=extent,
                   cmap='magma', vmin=-80, vmax=0)
    ax.set_title(f'2. Сглаженный спектр (окно {w}) — то, что вычитается')
    ax.set_ylabel('Частота (Гц)')
    fig.colorbar(im, ax=ax, label='dB', fraction=0.025)

    # Whitened
    ax = axes[2]
    v = 12
    im = ax.imshow(S_whiten, aspect='auto', origin='lower', extent=extent,
                   cmap='RdBu_r', vmin=-v, vmax=v)
    ax.set_title('3. Whitened = (1) − (2) — АЧХ тракта убрана, остались пики')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Частота (Гц)')
    fig.colorbar(im, ax=ax, label='дБ над локальной базой', fraction=0.025)

    fig.tight_layout()
    _maybe_show(save)


# ─── 4. delta histogram (главный график) ────────────────────────────────

def plot_delta(query_file: str, db: str = 'fingerprints.db',
               save: Optional[str] = None, top_n: int = 3):
    cfg = Config()
    with Engine(db, cfg) as eng:
        y, sr = load_audio(query_file, cfg.target_sr)
        hashes = eng.fingerprinter.from_samples(y)

        # Считаем votes — то же, что делает Matcher.match, но достаём наружу
        query_by_hash = defaultdict(list)
        for h, off in hashes:
            query_by_hash[h].append(off)

        votes = defaultdict(lambda: defaultdict(int))
        c = eng.index.conn.cursor()
        hash_list = list(query_by_hash.keys())
        for i in range(0, len(hash_list), 500):
            chunk = hash_list[i:i + 500]
            ph = ','.join('?' * len(chunk))
            rows = c.execute(
                f'SELECT hash, track_id, offset FROM fingerprints '
                f'WHERE hash IN ({ph})', chunk
            ).fetchall()
            for h, tid, db_off in rows:
                for q_off in query_by_hash[h]:
                    votes[tid][db_off - q_off] += 1

        if not votes:
            print('Нет ни одного совпадения — нечего рисовать', file=sys.stderr)
            return

        # Топ-N треков по лучшему пику
        ranked = []
        for tid, hist in votes.items():
            best_delta, best_score = max(hist.items(), key=lambda kv: kv[1])
            ranked.append((best_score, tid, best_delta))
        ranked.sort(reverse=True)
        top = ranked[:top_n]

        # Рисуем
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                 gridspec_kw={'height_ratios': [2, 1]})

        # Верх: полные гистограммы топ-3
        ax = axes[0]
        colors = ['lime', 'orange', 'cyan']
        for i, (score, tid, best_delta) in enumerate(top):
            title = eng.index.get_title(tid) or f'track_{tid}'
            hist = votes[tid]
            deltas = np.array(sorted(hist.keys()))
            counts = np.array([hist[d] for d in deltas])
            delta_sec = deltas * cfg.frame_duration
            label = f'{title[:40]}  peak={score}'
            ax.bar(delta_sec, counts, width=cfg.frame_duration * 0.9,
                   alpha=0.55, color=colors[i % len(colors)],
                   label=label)
            # Отметим вершину
            ax.axvline(best_delta * cfg.frame_duration,
                       color=colors[i % len(colors)],
                       linestyle='--', alpha=0.4)

        ax.set_xlabel('delta — позиция совпадения в треке (с)')
        ax.set_ylabel('Голосов')
        ax.set_title(
            f'Гистограмма сдвигов для {Path(query_file).name}  '
            f'({len(hashes)} хешей запроса). '
            f'Резкий пик = матч. Размазанные столбики = шум.'
        )
        ax.legend(loc='upper right', framealpha=0.4)
        ax.grid(alpha=0.2)

        # Низ: только победитель, крупно
        winner_score, winner_tid, winner_delta = top[0]
        winner_title = eng.index.get_title(winner_tid) or f'track_{winner_tid}'
        ax = axes[1]
        hist = votes[winner_tid]
        deltas = np.array(sorted(hist.keys()))
        counts = np.array([hist[d] for d in deltas])
        delta_sec = deltas * cfg.frame_duration
        ax.bar(delta_sec, counts, width=cfg.frame_duration * 0.9,
               color='lime', alpha=0.85)
        ax.axvline(winner_delta * cfg.frame_duration,
                   color='white', linestyle='--', alpha=0.7)
        ax.set_xlabel('delta (с)')
        ax.set_ylabel('Голосов')
        ax.set_title(f'Победитель: {winner_title}  '
                     f'(delta = {winner_delta * cfg.frame_duration:+.2f} с, '
                     f'score = {winner_score})')
        ax.grid(alpha=0.2)

        fig.tight_layout()
        _maybe_show(save)


# ─── 5. hashes: распределение ───────────────────────────────────────────

def plot_hashes(file: str, save: Optional[str] = None):
    cfg = Config()
    fp = Fingerprinter(cfg)
    y, sr = _load(file, cfg)
    hashes = fp.from_samples(y)

    if not hashes:
        print('Нет хешей — нечего рисовать', file=sys.stderr)
        return

    h_vals = np.array([h for h, _ in hashes], dtype=np.uint64)
    offsets = np.array([off for _, off in hashes])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Верх: распределение значений хеша
    ax = axes[0]
    # Логарифмические бины по значению хеша
    ax.hist(h_vals, bins=80, color='#00d4aa', alpha=0.8)
    ax.set_xlabel('Значение хеша')
    ax.set_ylabel('Частота')
    ax.set_title(f'Распределение хешей: {len(h_vals)} штук, '
                 f'{len(np.unique(h_vals))} уникальных '
                 f'({100 * len(np.unique(h_vals)) / len(h_vals):.0f}%)')
    ax.grid(alpha=0.2)

    # Низ: хеши во времени
    ax = axes[1]
    ax.scatter(offsets * cfg.frame_duration, h_vals % 10000,
               s=1, alpha=0.3, c='#ffaa00')
    ax.set_xlabel('Время в треке (с)')
    ax.set_ylabel('hash mod 10000')
    ax.set_title('Хеши по времени — должны быть равномерны')
    ax.grid(alpha=0.2)

    fig.tight_layout()
    _maybe_show(save)


# ─── 6. all ─────────────────────────────────────────────────────────────

def plot_all(file: str, db: str = 'fingerprints.db',
             save_prefix: Optional[str] = None):
    stem = Path(file).stem

    def _save(name):
        if save_prefix is None:
            return None
        return f'{save_prefix}_{name}.png'

    print('1/5 waveform...')
    plot_waveform(file, save=_save('waveform'))
    print('2/5 spectrogram + peaks...')
    plot_spectrogram(file, save=_save('spectrogram'))
    print('3/5 whiten...')
    plot_whiten(file, save=_save('whiten'))
    print('4/5 hashes...')
    plot_hashes(file, save=_save('hashes'))
    print('5/5 delta...')
    plot_delta(file, db, save=_save('delta'))


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    _setup_style()

    # Общие опции для всех подкоманд
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--db', default='fingerprints.db',
                        help='путь к SQLite базе (для delta)')
    common.add_argument('--save', default=None,
                        help='сохранить в PNG вместо показа '
                             '(для all — префикс)')

    p = argparse.ArgumentParser(
        prog='visualize',
        description='Визуализация работы auricle-core',
        parents=[common],
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    for name, help_text in [
        ('waveform',    'форма волны'),
        ('spectrogram', 'спектрограмма + пики'),
        ('whiten',      'сравнение до/после whitening'),
        ('hashes',      'распределение хешей'),
    ]:
        sp = sub.add_parser(name, help=help_text, parents=[common])
        sp.add_argument('file')

    pd = sub.add_parser('delta', help='гистограмма сдвигов для запроса',
                        parents=[common])
    pd.add_argument('file')
    pd.add_argument('--top-n', type=int, default=3,
                    help='сколько треков показывать (по умолчанию 3)')

    pa = sub.add_parser('all', help='все графики для одного файла',
                        parents=[common])
    pa.add_argument('file')

    args = p.parse_args()

    try:
        if args.cmd == 'waveform':
            plot_waveform(args.file, args.save)
        elif args.cmd == 'spectrogram':
            plot_spectrogram(args.file, args.save)
        elif args.cmd == 'whiten':
            plot_whiten(args.file, args.save)
        elif args.cmd == 'hashes':
            plot_hashes(args.file, args.save)
        elif args.cmd == 'delta':
            plot_delta(args.file, args.db, args.save, top_n=args.top_n)
        elif args.cmd == 'all':
            plot_all(args.file, args.db, save_prefix=args.save)
    except FileNotFoundError as e:
        print(f'Файл не найден: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Ошибка: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()