"""Единая конфигурация ядра. Меняешь тут — меняется везде."""
from dataclasses import dataclass

@dataclass
class Config:
    # --- аудио ---
    target_sr: int = 8000

    # --- STFT ---
    n_fft: int = 1024
    hop: int = 512
    window: str = 'hann'

    # --- пики ---
    peak_neighborhood: int = 15
    peak_threshold_db: float = -50.0     # для случая whiten=False
    whiten_threshold_db: float = 3.0     # для случая whiten=True (пик над базовой линией)

    # --- spectral whitening ---
    whiten: bool = True
    whiten_window: int = 41              # окно сглаживания по частоте (в бинах)

    # --- лог-частотные бины ---
    log_f_min: float = 40.0              # Гц, нижняя граница полезной полосы
    log_f_max: float = 4000.0            # Гц, верхняя (определяется target_sr/2)
    log_n_bins: int = 256                # 8 бит

    # --- хеши ---
    dt_min: int = 1
    dt_max: int = 63                     # ~4 сек при hop=512, sr=8000
    max_targets: int = 5
    mag_n_buckets: int = 16              # 4 бита на магнитуду

    # --- поиск ---
    min_score: int = 5
    top_k: int = 50

    @property
    def frame_duration(self) -> float:
        return self.hop / self.target_sr