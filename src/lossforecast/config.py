"""Settings shared by every module. The defaults are the values used in the notebook (Cell 1)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    min_assets: float = 1_000_000          # ratio features are left blank below this asset size
    max_days_to_file: int = 365            # later filings are catch-up filings and are excluded
    gap_range: tuple = (75, 110)           # days between two consecutive quarter-ends
    window_start: str = "2022-12-31"       # first quarter-end a regular filer can have in the 14 quarter files
    train_end: str = "2024-12-31"          # label quarters up to here are TRAIN
    valid_end: str = "2025-09-30"          # then VALIDATION up to here; later label quarters are TEST
    analysis_last_quarter: str = "2026q2"  # newer quarter files are kept aside for the single-use final test
    seed: int = 42
