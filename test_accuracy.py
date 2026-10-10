"""Прогоняет каждый трек папки через query, считает top-1 accuracy."""
import subprocess
import tempfile
from pathlib import Path

from auricle import Config, Engine

WAV_DIR = Path('wav')
QUERY_DURATION = 10
QUERY_OFFSETS = [10, 40, 80, 120, 160]  # секунды от начала


def get_duration(path: Path) -> float:
    """Длительность в секундах через ffprobe."""
    r = subprocess.run([
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(path),
    ], capture_output=True, text=True)
    return float(r.stdout.strip())


def main():
    cfg = Config()
    with Engine('fingerprints.db', cfg) as eng:
        tracks = sorted(WAV_DIR.glob('*.wav'))
        total = 0
        correct = 0
        misses = 0
        wrong = 0

        for track in tracks:
            expected = track.stem
            duration = get_duration(track)

            for offset in QUERY_OFFSETS:
                # пропускаем позиции, где не хватает 10 секунд до конца
                if offset + QUERY_DURATION > duration:
                    continue

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name

                r = subprocess.run([
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-i', str(track), '-ss', str(offset), '-t', str(QUERY_DURATION),
                    '-ar', '44100', '-ac', '2', tmp_path,
                ])
                if r.returncode != 0:
                    Path(tmp_path).unlink(missing_ok=True)
                    continue

                matches = eng.search_file(tmp_path)
                total += 1

                if matches and matches[0].title == expected:
                    correct += 1
                    m = matches[0]
                    print(f'OK   {expected[:30]:30} @{offset:4}s  '
                          f'score={m.score:4}  cov={m.coverage:.0%}  '
                          f'delta={m.delta * cfg.frame_duration:+.1f}s')
                elif matches:
                    wrong += 1
                    print(f'WRONG {expected[:30]:29} @{offset:4}s  '
                          f'-> {matches[0].title}  score={matches[0].score}')
                else:
                    misses += 1
                    print(f'MISS {expected[:30]:30} @{offset:4}s  (нет совпадений)')

                Path(tmp_path).unlink(missing_ok=True)

        print('-' * 70)
        if total:
            print(f'Top-1 accuracy: {correct}/{total} = {correct / total:.1%}')
            print(f'  OK:    {correct}')
            print(f'  MISS:  {misses}  (нет кандидатов)')
            print(f'  WRONG: {wrong}   (нашёл не тот трек)')


if __name__ == '__main__':
    main()