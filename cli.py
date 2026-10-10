"""Командная строка поверх ядра."""
import argparse
import sys
import time
from pathlib import Path

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from auricle import Config, Engine

DEFAULT_EXTS = ('wav',)


def _fmt_match(m, frame_duration):
    return (f'{m.title}  score={m.score}  '
            f'coverage={m.coverage:.1%}  '
            f'delta={m.delta} кадров ({m.delta * frame_duration:+.2f}с)')


def _fmt_eta(elapsed, done, total):
    if done == 0:
        return '—'
    per_item = elapsed / done
    remaining = per_item * (total - done)
    if remaining < 60:
        return f'{remaining:.0f}с'
    if remaining < 3600:
        return f'{remaining / 60:.1f}мин'
    return f'{remaining / 3600:.1f}ч'


def _collect_files(folder: Path, exts, recursive=True):
    """Возвращает отсортированный список файлов с нужными расширениями."""
    pattern = '**/*' if recursive else '*'
    exts = tuple('.' + e.lower().lstrip('.') for e in exts)
    files = [p for p in folder.glob(pattern)
             if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


# ==================== команды ====================

def cmd_index(args):
    cfg = Config()
    with Engine(args.db, cfg) as eng:
        t0 = time.time()
        total = 0
        for p in args.files:
            path = Path(p)
            title = args.title or path.stem
            try:
                t1 = time.time()
                tid = eng.add_file(title, str(path))
                total += 1
                print(f'  [{tid}] {title}  ({time.time() - t1:.2f}s)')
            except Exception as e:
                print(f'  skip {path}: {e}', file=sys.stderr)
        s = eng.stats()
        print(f'Всего: {total} файлов, {s["fingerprints"]} хешей, '
              f'{time.time() - t0:.2f}s')


# ─── parallel workers ──────────────────────────────────────────────────
# Эти функции выполняются в отдельных процессах. Модуль cli.py
# ре-импортируется в каждом воркере (Windows spawn), поэтому всё,
# что нужно — должно быть на уровне модуля.

_worker_fp = None


def _worker_init(cfg):
    """Вызывается один раз на воркер. Готовит Fingerprinter."""
    global _worker_fp
    from auricle.fingerprint import Fingerprinter
    _worker_fp = Fingerprinter(cfg)


def _worker_fingerprint(path_str, title):
    """Возвращает (title, hashes). Выполняется в subprocess."""
    hashes = _worker_fp.from_file(path_str)
    return title, hashes

def cmd_index_dir(args):
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f'Не папка: {folder}', file=sys.stderr)
        sys.exit(1)

    exts = args.ext.split(',') if args.ext else DEFAULT_EXTS
    files = _collect_files(folder, exts, recursive=not args.flat)

    if not files:
        print(f'В {folder} не найдено файлов с расширениями {exts}')
        return

    if args.limit:
        files = files[:args.limit]

    with Engine(args.db) as eng:
        existing = {title for _, title in eng.list_tracks()}

        to_do = []
        skipped = 0
        for p in files:
            title = p.stem if not args.prefix_subdir else str(
                p.relative_to(folder).with_suffix(''))
            if args.skip_existing and title in existing:
                skipped += 1
                continue
            to_do.append((p, title))

        print(f'Найдено файлов: {len(files)}  |  '
              f'пропущено (уже в базе): {skipped}  |  '
              f'к обработке: {len(to_do)}')
        if not to_do:
            print('Нечего делать.')
            return

        print(f'Расширения: {", ".join(exts)}  |  '
              f'рекурсивно: {"нет" if args.flat else "да"}')
        print('-' * 70)

        t0 = time.time()
        done = 0
        failed = 0
        workers = args.workers if args.workers and args.workers > 0 \
            else (os.cpu_count() or 1)

        if workers <= 1:
            print('Режим: последовательный (1 поток)')
            for i, (p, title) in enumerate(to_do, 1):
                rel = p.relative_to(folder)
                try:
                    t1 = time.time()
                    tid = eng.add_file(title, str(p))
                    dt = time.time() - t1
                    done += 1
                    eta = _fmt_eta(time.time() - t0, i, len(to_do))
                    print(f'  [{i}/{len(to_do)}] [{tid}] {rel}  '
                          f'({dt:.2f}s, ETA {eta})')
                except Exception as e:
                    failed += 1
                    print(f'  [{i}/{len(to_do)}] FAIL {rel}: {e}',
                          file=sys.stderr)
        else:
            print(f'Режим: параллельный ({workers} воркеров)')
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(eng.config,),
            ) as pool:
                futures = {
                    pool.submit(_worker_fingerprint, str(p), title): (p, title)
                    for p, title in to_do
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    p, title = futures[fut]
                    rel = p.relative_to(folder)
                    try:
                        got_title, hashes = fut.result()
                        t1 = time.time()
                        tid = eng.index.add(got_title, hashes)
                        dt = time.time() - t1
                        done += 1
                        eta = _fmt_eta(time.time() - t0, i, len(to_do))
                        print(f'  [{i}/{len(to_do)}] [{tid}] {rel}  '
                              f'({dt:.2f}s write, ETA {eta})')
                    except Exception as e:
                        failed += 1
                        print(f'  [{i}/{len(to_do)}] FAIL {rel}: {e}',
                              file=sys.stderr)

        s = eng.stats()
        total_dt = time.time() - t0
        print('-' * 70)
        print(f'Готово: {done} добавлено, {failed} ошибок, '
              f'{skipped} пропущено за {total_dt:.1f}s')
        print(f'Всего в базе: {s["tracks"]} треков, '
              f'{s["fingerprints"]} хешей')


def cmd_query(args):
    with Engine(args.db) as eng:
        try:
            matches = eng.search_file(args.file)
        except Exception as e:
            print(f'Ошибка: {e}', file=sys.stderr)
            sys.exit(1)
        if not matches:
            print('Не найдено')
            return
        for m in matches:
            print(_fmt_match(m, eng.config.frame_duration))


def cmd_list(args):
    with Engine(args.db) as eng:
        for tid, title in eng.list_tracks():
            print(f'  [{tid}] {title}')
        s = eng.stats()
        print(f'Итого: {s["tracks"]} треков, {s["fingerprints"]} хешей')


def cmd_delete(args):
    with Engine(args.db) as eng:
        eng.remove_track(args.track_id)
        print(f'Удалён трек {args.track_id}')


# ==================== точка входа ====================

def main():
    p = argparse.ArgumentParser(prog='fp')
    p.add_argument('--db', default='fingerprints.db')
    sub = p.add_subparsers(dest='cmd', required=True)

    # --- index: конкретные файлы ---
    pi = sub.add_parser('index', help='проиндексировать отдельные файлы')
    pi.add_argument('files', nargs='+')
    pi.add_argument('--title', help='имя (если один файл)')
    pi.set_defaults(func=cmd_index)

    # --- index-dir: вся папка ---
    pid = sub.add_parser('index-dir', help='проиндексировать папку рекурсивно')
    pid.add_argument('folder', help='путь к папке')
    pid.add_argument('--ext', default='wav',
                     help='расширения через запятую (по умолчанию: wav). '
                          'Пример: --ext wav,flac,mp3')
    pid.add_argument('--flat', action='store_true',
                     help='не заходить в подпапки')
    pid.add_argument('--limit', type=int, default=0,
                     help='обработать только первые N файлов (0 = все)')
    pid.add_argument('--no-skip-existing', dest='skip_existing',
                     action='store_false',
                     help='переиндексировать даже те, что уже в базе')
    pid.add_argument('--prefix-subdir', action='store_true',
                     help='включать подпапку в title (помогает против коллизий '
                          'типа Disc1/01.wav)')
    pid.add_argument('--workers', type=int, default=0,
                 help='число параллельных воркеров '
                      '(0 = авто = кол-во ядер, 1 = без параллелизма)')
    pid.set_defaults(func=cmd_index_dir, skip_existing=True)

    # --- query ---
    pq = sub.add_parser('query', help='найти трек по отрывку')
    pq.add_argument('file')
    pq.set_defaults(func=cmd_query)

    # --- list ---
    pl = sub.add_parser('list', help='список треков в базе')
    pl.set_defaults(func=cmd_list)

    # --- delete ---
    pd = sub.add_parser('delete', help='удалить трек по id')
    pd.add_argument('track_id', type=int)
    pd.set_defaults(func=cmd_delete)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()