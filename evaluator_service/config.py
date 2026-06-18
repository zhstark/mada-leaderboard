from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    work_dir: Path
    max_download_bytes: int
    max_uncompressed_bytes: int
    max_zip_members: int
    download_timeout_seconds: float
    retain_work_dir: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            work_dir=Path(os.getenv("EVALUATOR_WORK_DIR", PROJECT_ROOT / "var" / "evaluator_jobs")),
            max_download_bytes=int(os.getenv("EVALUATOR_MAX_DOWNLOAD_BYTES", str(250 * 1024 * 1024))),
            max_uncompressed_bytes=int(
                os.getenv("EVALUATOR_MAX_UNCOMPRESSED_BYTES", str(750 * 1024 * 1024))
            ),
            max_zip_members=int(os.getenv("EVALUATOR_MAX_ZIP_MEMBERS", "20000")),
            download_timeout_seconds=float(os.getenv("EVALUATOR_DOWNLOAD_TIMEOUT_SECONDS", "120")),
            retain_work_dir=os.getenv("EVALUATOR_RETAIN_WORK_DIR", "1").lower()
            not in {"0", "false", "no"},
        )


def ground_truth_dir_for_topic(topic_id: int) -> Path:
    if topic_id == 1:
        return PROJECT_ROOT / "q1" / "labels"
    if topic_id == 2:
        return PROJECT_ROOT / "q2" / "labels"
    raise ValueError(f"Unsupported topic_id: {topic_id}")
