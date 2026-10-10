"""Index: SQLite-хранилище хешей. Режим «пополнение базы»."""
import sqlite3
from typing import List, Optional, Tuple

from .config import Config
from .fingerprint import Hash

_SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL UNIQUE
    )''',
    '''CREATE TABLE IF NOT EXISTS fingerprints (
        hash INTEGER NOT NULL,
        track_id INTEGER NOT NULL,
        offset INTEGER NOT NULL
    )''',
    'CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints(hash)',
    'CREATE INDEX IF NOT EXISTS idx_track ON fingerprints(track_id)',
]


class Index:
    def __init__(self, path: str, config: Config):
        self.config = config
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        for stmt in _SCHEMA:
            self.conn.execute(stmt)
        self.conn.commit()

    # --- запись ---

    def add(self, title: str, hashes: List[Hash]) -> int:
        """Добавляет трек. Если title уже есть — переиндексирует."""
        c = self.conn.cursor()
        row = c.execute('SELECT id FROM tracks WHERE title = ?', (title,)).fetchone()
        if row:
            track_id = row[0]
            c.execute('DELETE FROM fingerprints WHERE track_id = ?', (track_id,))
        else:
            c.execute('INSERT INTO tracks (title) VALUES (?)', (title,))
            track_id = c.lastrowid
        c.executemany(
            'INSERT INTO fingerprints (hash, track_id, offset) VALUES (?, ?, ?)',
            [(h, track_id, off) for h, off in hashes],
        )
        self.conn.commit()
        return track_id

    def remove(self, track_id: int) -> None:
        self.conn.execute('DELETE FROM fingerprints WHERE track_id = ?', (track_id,))
        self.conn.execute('DELETE FROM tracks WHERE id = ?', (track_id,))
        self.conn.commit()

    # --- чтение ---

    def get_title(self, track_id: int) -> Optional[str]:
        row = self.conn.execute(
            'SELECT title FROM tracks WHERE id = ?', (track_id,)
        ).fetchone()
        return row[0] if row else None

    def list_tracks(self) -> List[Tuple[int, str]]:
        return self.conn.execute(
            'SELECT id, title FROM tracks ORDER BY id'
        ).fetchall()

    def stats(self) -> dict:
        tracks = self.conn.execute('SELECT COUNT(*) FROM tracks').fetchone()[0]
        fps = self.conn.execute('SELECT COUNT(*) FROM fingerprints').fetchone()[0]
        return {'tracks': tracks, 'fingerprints': fps}

    # --- жизненный цикл ---

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()