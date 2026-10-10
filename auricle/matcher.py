"""Matcher: поиск совпадения. Режим «работа с базой»."""
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from .audio import load_audio
from .config import Config
from .fingerprint import Fingerprinter, Hash
from .index import Index

_CHUNK = 500  # сколько хешей в одном SQL-запросе (обход лимита параметров)


@dataclass
class Match:
    """Результат поиска. Совместим с live-режимом:

    delta — сдвиг между запросом и треком в базе, в кадрах STFT.
    Если запрос — фрагмент трека X, начиная с секунды 15 (при hop=512, sr=8000
    это кадр 15*8000/512 ≈ 234), то delta = -234 (db_offset меньше query_offset).

    Для live: при непрерывном воспроизведении одного трека delta монотонно
    убывает (или растёт — зависит от знака) пропорционально времени. Резкий
    скачок delta — признак смены трека.
    """
    track_id: int
    title: str
    score: int
    delta: int
    query_hashes: int
    coverage: float  # score / query_hashes, 0..1

    @property
    def delta_seconds(self) -> float:
        # длительность кадра нужно знать снаружи; оставим метод для удобства
        raise AttributeError('используй engine.config.frame_duration')


class Matcher:
    def __init__(self, index: Index, fingerprinter: Fingerprinter):
        self.index = index
        self.fp = fingerprinter
        self.config = fingerprinter.config

    # --- публичное API ---

    def match_file(self, path: str) -> List[Match]:
        hashes = self.fp.from_file(path)
        return self.match(hashes)

    def match_samples(self, y) -> List[Match]:
        hashes = self.fp.from_samples(y)
        return self.match(hashes)

    def match(self, hashes: List[Hash], top_k: Optional[int] = None) -> List[Match]:
        if not hashes:
            return []
        top_k = top_k or self.config.top_k

        # группируем запрос по хешу: один хеш может встречаться в разных местах
        query_by_hash = defaultdict(list)
        for h, off in hashes:
            query_by_hash[h].append(off)

        # votes[track_id][delta] = сколько раз совпал с этим сдвигом
        votes = defaultdict(lambda: defaultdict(int))

        c = self.index.conn.cursor()
        hash_list = list(query_by_hash.keys())
        for i in range(0, len(hash_list), _CHUNK):
            chunk = hash_list[i:i + _CHUNK]
            ph = ','.join('?' * len(chunk))
            rows = c.execute(
                f'SELECT hash, track_id, offset FROM fingerprints WHERE hash IN ({ph})',
                chunk,
            ).fetchall()
            for h, tid, db_off in rows:
                for q_off in query_by_hash[h]:
                    votes[tid][db_off - q_off] += 1

        # собираем ВСЕ (track, delta) с достаточным score — не только топ-1 на трек
        candidates = []
        for tid, hist in votes.items():
            for delta, score in hist.items():
                if score >= self.config.min_score:
                    candidates.append((score, tid, delta))
        candidates.sort(reverse=True)

        results: List[Match] = []
        n_q = len(hashes)
        for score, tid, delta in candidates[:top_k]:
            if score < self.config.min_score:
                break
            title = self.index.get_title(tid)
            if title is None:
                continue
            results.append(Match(
                track_id=tid,
                title=title,
                score=score,
                delta=delta,
                query_hashes=n_q,
                coverage=score / n_q,
            ))
        return results