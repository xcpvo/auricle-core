"""Live-детектор: слушает поток и ведёт таймлайн распознанных треков.

Режимы:
  python live.py simulate wav/Зависим.wav              # эмуляция из файла
  python live.py simulate wav/Зависим.wav --realtime   # с реальным темпом
  python live.py listen --device 77 --debug            # с микрофона + диагностика
  python live.py listen --device 73 --debug            # с loopback Voicemeeter
  python live.py devices                               # список аудиоустройств
"""
import argparse
import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from auricle import Config, Engine
from auricle.audio import load_audio
from auricle.matcher import Match


# ─── настройки live ────────────────────────────────────────────────────
WINDOW_SEC = 10          # длина окна фингерпринта
HOP_SEC = 2              # шаг между окнами
MIN_SCORE_START = 25      # для мгновенного старта, если матч сильный
MIN_SCORE_HISTORY = 8     # минимальный score, чтобы попасть в историю
MIN_CONSECUTIVE = 3       # сколько окон подряд для подтверждения по истории
HISTORY_SIZE = 6          # сколько окон держим в истории
MAX_MISSES = 4            # сколько пустых окон = трек закончился
SWITCH_RATIO = 1.5       # во сколько раз новый трек должен быть сильнее
DELTA_TOLERANCE = 5      # допуск прироста delta (в кадрах)


# ─── кольцевой буфер ──────────────────────────────────────────────────

class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buf = np.zeros(size, dtype=np.float32)
        self.pos = 0
        self.filled = 0

    def write(self, chunk: np.ndarray) -> None:
        n = len(chunk)
        if n >= self.size:
            self.buf[:] = chunk[-self.size:]
            self.pos = 0
            self.filled = self.size
            return
        end = self.pos + n
        if end <= self.size:
            self.buf[self.pos:end] = chunk
        else:
            split = self.size - self.pos
            self.buf[self.pos:] = chunk[:split]
            self.buf[:end - self.size] = chunk[split:]
        self.pos = end % self.size
        self.filled = min(self.size, self.filled + n)

    def read_all(self) -> Optional[np.ndarray]:
        if self.filled < self.size:
            return None
        if self.pos == 0:
            return self.buf.copy()
        return np.concatenate([self.buf[self.pos:], self.buf[:self.pos]])


# ─── state machine ────────────────────────────────────────────────────

@dataclass
class TrackState:
    track_id: int
    title: str
    start_wall: datetime
    last_wall: datetime
    last_delta: int
    last_score: int
    misses: int = 0

    def duration(self) -> float:
        return (self.last_wall - self.start_wall).total_seconds()


class LiveTracker:
    def __init__(self, engine: Engine, debug: bool = False):
        self.eng = engine
        self.cfg = engine.config
        self.debug = debug
        self.current: Optional[TrackState] = None
        self.history: list = []
        self._windows_processed = 0
        self._recent = []  # последние HISTORY_SIZE окон: (track_id, title, delta, score)

    def process_window(self, y: np.ndarray) -> None:
        raw_peak = float(np.max(np.abs(y)))
        raw_rms = float(np.sqrt(np.mean(y * y)))

        if raw_peak > 1e-6:
            y = y / raw_peak * 0.9

        matches = self.eng.search_samples(y)
        now = datetime.now()
        self._windows_processed += 1
        top = matches[0] if matches else None

        # --- диагностика ---
        if self.debug:
            top_str = (f'{top.title[:22]:22} score={top.score:4} '
                       f'delta={top.delta * self.cfg.frame_duration:+7.1f}s '
                       f'cov={top.coverage:.0%}'
                       ) if top else '(нет матчей)'
            marker = ''
            if self.current is not None:
                marker = f' [tracking: {self.current.title[:18]}]'
            print(f'  [dbg #{self._windows_processed:3}] '
                  f'rms={raw_rms:.4f} peak={raw_peak:.4f}  {top_str}{marker}')

        # --- добавляем в историю ---
        if top is not None and top.score >= MIN_SCORE_HISTORY:
            self._recent.append({
                'track_id': top.track_id,
                'title': top.title,
                'delta': top.delta,
                'score': top.score,
            })
        else:
            self._recent.append(None)
        if len(self._recent) > HISTORY_SIZE:
            self._recent.pop(0)

        # --- ищем кандидата в истории ---
        candidate = self._find_candidate()

        # ─── трека ещё нет ───
        if self.current is None:
            # либо мгновенный сильный матч, либо стабильный слабый
            if top is not None and top.score >= MIN_SCORE_START:
                self._start_now(top, now)
                return
            if candidate is not None:
                self._start_from_candidate(candidate, now)
            return

        cur = self.current
        expected = cur.last_delta + int(HOP_SEC / self.cfg.frame_duration)

        # ─── продолжаем текущий: ищем его в истории ───
        same_recent = [
            h for h in self._recent
            if h is not None and h['track_id'] == cur.track_id
        ]
        if same_recent:
            best = min(same_recent, key=lambda h: abs(h['delta'] - expected))
            # если delta близка — продолжаем
            if abs(best['delta'] - expected) <= 3 * DELTA_TOLERANCE:
                cur.last_delta = best['delta']
                cur.last_wall = now
                cur.last_score = max(cur.last_score, best['score'])
                cur.misses = 0
                return
            # если delta далеко — но трек тот же — считаем как слабый continue
            cur.last_delta = best['delta']
            cur.last_wall = now
            cur.misses = 0
            return

        # ─── своего трека нет — miss или смена ───
        if candidate is not None and candidate['track_id'] != cur.track_id:
            if candidate['avg_score'] > cur.last_score * SWITCH_RATIO:
                self._on_end(now)
                self._start_from_candidate(candidate, now)
                return

        self._on_miss(now)

    def _find_candidate(self):
        """Ищет трек, который стабильно появляется в последних окнах
        с согласованной delta.

        Возвращает {'track_id', 'title', 'avg_score', 'last_delta'} или None.
        """
        if len(self._recent) < MIN_CONSECUTIVE:
            return None

        expected_advance = int(HOP_SEC / self.cfg.frame_duration)

        # Группируем по track_id
        stats = {}
        for h in self._recent:
            if h is None:
                continue
            tid = h['track_id']
            if tid not in stats:
                stats[tid] = {'title': h['title'], 'scores': [], 'deltas': [], 'idx': []}
            stats[tid]['scores'].append(h['score'])
            stats[tid]['deltas'].append(h['delta'])
            stats[tid]['idx'].append(self._recent.index(h))

        best = None
        for tid, s in stats.items():
            if len(s['scores']) < MIN_CONSECUTIVE:
                continue

            # проверяем согласованность delta: между последовательными окнами
            # delta должна расти примерно на expected_advance
            deltas = s['deltas']
            consistent = True
            for i in range(1, len(deltas)):
                advance = deltas[i] - deltas[i-1]
                if abs(advance - expected_advance) > 3 * DELTA_TOLERANCE:
                    consistent = False
                    break

            if not consistent:
                continue

            avg_score = sum(s['scores']) / len(s['scores'])
            if best is None or avg_score > best['avg_score']:
                best = {
                    'track_id': tid,
                    'title': s['title'],
                    'avg_score': int(avg_score),
                    'last_delta': deltas[-1],
                }
        return best

    def _start_now(self, best: Match, now):
        self.current = TrackState(
            track_id=best.track_id,
            title=best.title,
            start_wall=now,
            last_wall=now,
            last_delta=best.delta,
            last_score=best.score,
        )
        self._emit_start(self.current, best.score)

    def _start_from_candidate(self, candidate, now):
        self.current = TrackState(
            track_id=candidate['track_id'],
            title=candidate['title'],
            start_wall=now,
            last_wall=now,
            last_delta=candidate['last_delta'],
            last_score=candidate['avg_score'],
        )
        self._emit_start(self.current, candidate['avg_score'])

    def _on_miss(self, now):
        if self.current is not None:
            self.current.misses += 1
            if self.current.misses >= MAX_MISSES:
                self._on_end(now)

    def _on_end(self, now):
        if self.current is None:
            return
        cur = self.current
        self.history.append(cur)
        self._emit_end(cur)
        self.current = None

    def _ts(self):
        return datetime.now().strftime('%H:%M:%S')

    def _emit_start(self, state: TrackState, score: int):
        print(f'[{self._ts()}]  START  {state.title}  score={score}')

    def _emit_end(self, state: TrackState):
        print(f'[{self._ts()}]  END    {state.title}  '
              f'duration={state.duration():.1f}s')


# ─── режим 1: симуляция из файла ──────────────────────────────────────

def run_simulate(engine: Engine, path: str, realtime: bool = False,
                 debug: bool = False):
    cfg = engine.config
    sr = cfg.target_sr
    hop_samples = HOP_SEC * sr

    y, _ = load_audio(path, sr)
    print(f'Эмулирую поток из {path} ({len(y) / sr:.1f}с аудио)')
    print(f'окно={WINDOW_SEC}с  шаг={HOP_SEC}с  realtime={realtime}')
    print('-' * 70)

    buf = RingBuffer(WINDOW_SEC * sr)
    tracker = LiveTracker(engine, debug=debug)

    t0 = time.time()
    for i in range(0, len(y), hop_samples):
        chunk = y[i:i + hop_samples]
        if len(chunk) < hop_samples:
            chunk = np.pad(chunk, (0, hop_samples - len(chunk)))
        buf.write(chunk)
        window = buf.read_all()
        if window is None:
            continue
        try:
            tracker.process_window(window)
        except Exception as e:
            print(f'  ERR: {e}', file=sys.stderr)

        if realtime:
            target = (i + hop_samples) / sr
            actual = time.time() - t0
            if target > actual:
                time.sleep(target - actual)

    if tracker.current is not None:
        tracker._on_end(datetime.now())

    print('-' * 70)
    total = len(tracker.history) + (1 if tracker.current else 0)
    print(f'Всего распознано треков: {total}')


# ─── режим 2: микрофон / loopback ─────────────────────────────────────

def run_microphone(engine: Engine, device: Optional[int] = None,
                   debug: bool = False):
    """Захват с микрофона/loopback.

    PortAudio требует нативную частоту устройства (44100/48000),
    поэтому захватываем в ней, а потом ресемплим в 8000 для ядра.
    """
    import sounddevice as sd
    from scipy.signal import resample_poly
    from math import gcd

    dev_info = sd.query_devices(device, 'input')
    native_sr = int(dev_info['default_samplerate'])
    max_ch = dev_info['max_input_channels']
    channels = min(2, max_ch) or 1

    target_sr = engine.config.target_sr
    g = gcd(native_sr, target_sr)
    up = target_sr // g
    down = native_sr // g
    hop_target = HOP_SEC * target_sr

    buf = RingBuffer(WINDOW_SEC * target_sr)
    tracker = LiveTracker(engine, debug=debug)
    q: queue.Queue = queue.Queue()
    leftover = np.zeros(0, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        if status:
            print(f'  audio status: {status}', file=sys.stderr)
        q.put(indata.copy())

    print(f'Устройство:     {dev_info["name"]}')
    print(f'Нативный sr:    {native_sr} Hz, каналов: {channels}')
    print(f'Ресемплинг:     {native_sr} -> {target_sr} (up={up}, down={down})')
    print(f'Окно:           {WINDOW_SEC}с, шаг: {HOP_SEC}с')
    if debug:
        print('Режим:          DEBUG (вывод уровня сигнала каждые 2с)')
    print('Ctrl+C — выход')
    print('-' * 70)

    with sd.InputStream(samplerate=native_sr, channels=channels,
                        dtype='float32', device=device,
                        callback=callback):
        while True:
            chunk = q.get()
            mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk
            resampled = resample_poly(mono, up, down).astype(np.float32)
            leftover = np.concatenate([leftover, resampled])

            while len(leftover) >= hop_target:
                piece = leftover[:hop_target]
                leftover = leftover[hop_target:]
                buf.write(piece)
                window = buf.read_all()
                if window is None:
                    continue
                try:
                    tracker.process_window(window)
                except Exception as e:
                    print(f'  ERR: {e}', file=sys.stderr)


def run_devices():
    import sounddevice as sd
    print(sd.query_devices())


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(prog='live')
    p.add_argument('--db', default='fingerprints.db')
    sub = p.add_subparsers(dest='mode', required=True)

    pl = sub.add_parser('listen', help='слушать микрофон/loopback')
    pl.add_argument('--device', type=int, default=None,
                    help='индекс устройства (см. python live.py devices)')
    pl.add_argument('--debug', action='store_true',
                    help='показывать rms/peak и лучший матч каждые 2с')

    ps = sub.add_parser('simulate', help='проиграть файл как поток')
    ps.add_argument('file')
    ps.add_argument('--realtime', action='store_true')
    ps.add_argument('--debug', action='store_true')

    pd = sub.add_parser('devices', help='список аудиоустройств')
    pd.set_defaults(_dummy=True)

    args = p.parse_args()

    if args.mode == 'devices':
        run_devices()
        return

    with Engine(args.db, Config()) as eng:
        if args.mode == 'listen':
            try:
                run_microphone(eng, args.device, debug=args.debug)
            except KeyboardInterrupt:
                print('\nОстановлено.')
        else:
            run_simulate(eng, args.file, args.realtime, debug=args.debug)


if __name__ == '__main__':
    main()