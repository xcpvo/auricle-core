"""Тест устойчивости ядра к деградации звука.

Каждый запрос — 10-секундный фрагмент трека, пропущенный через
ffmpeg-фильтр, имитирующий реальный сценарий (микрофон, радио, стрим).

Для каждого сценария считаем top-1 accuracy и средний score.
"""
import subprocess
import tempfile
from pathlib import Path

from auricle import Config, Engine

WAV_DIR = Path('wav')
QUERY_DURATION = 10
QUERY_OFFSETS = list(range(10, 160, 10))  # 10, 20, ..., 150


# ─── определения деградаций ────────────────────────────────────────────
# Каждый элемент: (имя, список аргументов для ffmpeg ПОСЛЕ -i/-ss/-t).
# Для сценариев mp3_* фильтры игнорируются, идёт двухшаговый кодек.

DEGRADATIONS = [
    ('clean',            []),

    # --- кодеки / битрейт (через настоящий mp3-файл, см. make_degraded) ---
    ('mp3_128',          None),
    ('mp3_64',           None),
    ('mp3_32',           None),

    # --- полосовые фильтры ---
    ('lowpass_4k',       ['-af', 'lowpass=f=4000']),
    ('lowpass_2k',       ['-af', 'lowpass=f=2000']),
    ('highpass_300',     ['-af', 'highpass=f=300']),
    ('bandpass_phone',   ['-af', 'highpass=f=300,lowpass=f=3400']),

    # --- шум (три уровня громкости) ---
    ('noise_light',      ['-af', 'aeval=val(0)+0.01*random(0)']),
    ('noise_medium',     ['-af', 'aeval=val(0)+0.03*random(0)']),
    ('noise_heavy',      ['-af', 'aeval=val(0)+0.08*random(0)']),

    # --- реверберация (большая комната) ---
    ('reverb',           ['-af', 'aecho=0.8:0.9:40|60:0.4|0.3']),

    # --- клиппинг (перегрузка микрофона) ---
    ('clipped',          ['-af', 'volume=6dB,alimiter=limit=0.5']),

    # --- питч/темп (кассета, винил) ---
    ('pitch_up_2pct',    ['-af', 'asetrate=45024,aresample=44100']),
    ('pitch_down_2pct',  ['-af', 'asetrate=43176,aresample=44100']),
    ('tempo_up_5pct',    ['-af', 'atempo=1.05']),

    # --- комбо: как радио или звонок ---
    ('radio_like',       ['-af', 'highpass=f=200,lowpass=f=5000,volume=1.5,alimiter=limit=0.9']),
]


def get_duration(path: Path) -> float:
    r = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(path),
    ], capture_output=True, text=True)
    return float(r.stdout.strip())


def _run_ffmpeg(cmd) -> bool:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'    ffmpeg fail: {r.stderr.strip().splitlines()[-1] if r.stderr else "?"}')
    return r.returncode == 0


def make_degraded(track: Path, offset: int, duration: int,
                  filters, out_path: str, name: str) -> bool:
    """Нарезает фрагмент и применяет деградацию.

    Для mp3_* — двухшаговый кодек: WAV → MP3 → WAV. Это эмулирует
    реальный стриминг (отдача mp3-потока, декодирование у клиента).

    Для остальных — одиночный ffmpeg-вызов с фильтрами.
    """
    # ─── MP3: двухшаговый кодек ───
    if name.startswith('mp3_'):
        bitrate = name.split('_', 1)[1] + 'k'   # '128k', '64k', '32k'
        mp3_path = out_path + '.mp3'

        # шаг 1: нарезаем и кодируем в mp3
        step1 = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', str(track), '-ss', str(offset), '-t', str(duration),
            '-c:a', 'libmp3lame', '-b:a', bitrate,
            '-ar', '44100', '-ac', '2', mp3_path,
        ]
        if not _run_ffmpeg(step1):
            Path(mp3_path).unlink(missing_ok=True)
            return False

        # шаг 2: декодируем mp3 обратно в wav
        step2 = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', mp3_path,
            '-ar', '44100', '-ac', '2', out_path,
        ]
        ok = _run_ffmpeg(step2)
        Path(mp3_path).unlink(missing_ok=True)
        return ok

    # ─── Обычные фильтры ───
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(track), '-ss', str(offset), '-t', str(duration),
    ]
    if filters:
        cmd.extend(filters)
    cmd.extend(['-ar', '44100', '-ac', '2', out_path])
    return _run_ffmpeg(cmd)


def main():
    cfg = Config()
    with Engine('fingerprints.db', cfg) as eng:
        tracks = sorted(WAV_DIR.glob('*.wav'))
        if not tracks:
            print(f'Нет WAV-файлов в {WAV_DIR}')
            return

        durations = {t: get_duration(t) for t in tracks}

        results = {name: {'ok': 0, 'miss': 0, 'wrong': 0, 'scores': []}
                   for name, _ in DEGRADATIONS}

        # Сколько запросов на сценарий ожидается — для прогресса
        planned = sum(
            1 for t in tracks for off in QUERY_OFFSETS
            if off + QUERY_DURATION <= durations[t]
        )
        total_planned = planned * len(DEGRADATIONS)
        done = 0

        print(f'Треков: {len(tracks)}, запросов на сценарий: {planned}, '
              f'всего прогонов: {total_planned}')
        print('-' * 78)

        for track in tracks:
            expected = track.stem
            duration = durations[track]

            for offset in QUERY_OFFSETS:
                if offset + QUERY_DURATION > duration:
                    continue

                for name, filters in DEGRADATIONS:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        tmp_path = tmp.name

                    ok = make_degraded(track, offset, QUERY_DURATION,
                                       filters, tmp_path, name)
                    done += 1
                    if done % 20 == 0 or done == total_planned:
                        print(f'  прогресс: {done}/{total_planned}', end='\r')

                    if not ok:
                        Path(tmp_path).unlink(missing_ok=True)
                        continue

                    try:
                        matches = eng.search_file(tmp_path)
                    except Exception as e:
                        print(f'\n  ERR {name} {expected} @{offset}: {e}')
                        Path(tmp_path).unlink(missing_ok=True)
                        continue

                    r = results[name]
                    if matches and matches[0].title == expected:
                        r['ok'] += 1
                        r['scores'].append(matches[0].score)
                    elif matches:
                        r['wrong'] += 1
                        r['scores'].append(matches[0].score)
                    else:
                        r['miss'] += 1
                        r['scores'].append(0)

                    Path(tmp_path).unlink(missing_ok=True)

        print(' ' * 40)  # очистка строки прогресса
        print('=' * 78)
        print(f'{"Сценарий":<20} {"OK":>4} {"MISS":>5} {"WRONG":>6} '
              f'{"Accuracy":>9} {"AvgScore":>9}')
        print('-' * 78)

        for name, _ in DEGRADATIONS:
            r = results[name]
            total = r['ok'] + r['miss'] + r['wrong']
            acc = r['ok'] / total if total else 0
            avg = sum(r['scores']) / len(r['scores']) if r['scores'] else 0
            print(f'{name:<20} {r["ok"]:>4} {r["miss"]:>5} {r["wrong"]:>6} '
                  f'{acc:>8.1%} {avg:>9.0f}')

        print('=' * 78)


if __name__ == '__main__':
    main()