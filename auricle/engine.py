"""Engine: единая точка входа. Ровно две «двери» — пополнение и поиск."""
from typing import List, Optional

from .config import Config
from .fingerprint import Fingerprinter
from .index import Index
from .matcher import Match, Matcher


class Engine:
    def __init__(self, db_path: str, config: Optional[Config] = None):
        self.config = config or Config()
        self.fingerprinter = Fingerprinter(self.config)
        self.index = Index(db_path, self.config)
        self.matcher = Matcher(self.index, self.fingerprinter)

    # ================= РЕЖИМ 1: пополнение базы =================

    def add_file(self, title: str, path: str) -> int:
        hashes = self.fingerprinter.from_file(path)
        return self.index.add(title, hashes)

    def add_samples(self, title: str, y) -> int:
        hashes = self.fingerprinter.from_samples(y)
        return self.index.add(title, hashes)

    def remove_track(self, track_id: int) -> None:
        self.index.remove(track_id)

    def list_tracks(self):
        return self.index.list_tracks()

    def stats(self) -> dict:
        return self.index.stats()

    # ================= РЕЖИМ 2: поиск совпадения =================

    def search_file(self, path: str) -> List[Match]:
        return self.matcher.match_file(path)

    def search_samples(self, y) -> List[Match]:
        return self.matcher.match_samples(y)

    def search_hashes(self, hashes) -> List[Match]:
        return self.matcher.match(hashes)

    # ================= жизненный цикл =================

    def close(self) -> None:
        self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()