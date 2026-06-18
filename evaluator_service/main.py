from __future__ import annotations

import logging
import shutil
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI

from .archive import download_zip, extract_zip_safely
from .config import Settings, ground_truth_dir_for_topic
from .database import SubmissionRepository
from .errors import FormatValidationError
from .labels import validate_prediction_content, validate_prediction_files
from .models import EvaluateRequest, EvaluateResponse
from .scoring import run_scoring_job


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MADA Competition Evaluator", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, object]:
    settings = Settings.from_env()
    return {
        "ok": True,
        "database_configured": bool(settings.database_url),
        "topics": [1, 2],
    }


@app.post("/submissions", response_model=EvaluateResponse)
async def create_submission(request: EvaluateRequest, background_tasks: BackgroundTasks) -> EvaluateResponse:
    settings = Settings.from_env()
    repository = SubmissionRepository(settings.database_url)
    job_id = str(uuid4())
    job_dir = settings.work_dir / job_id
    zip_path = job_dir / request.file_name
    extracted_dir = job_dir / "extracted"

    try:
        _validate_request_file_name(request.file_name)
        ground_truth_dir = ground_truth_dir_for_topic(request.topic_id)
        await download_zip(str(request.download_url), zip_path, request.file_size, settings)
        prediction_dir = extract_zip_safely(zip_path, extracted_dir, settings)
        prediction_files = validate_prediction_files(prediction_dir, ground_truth_dir)
        validate_prediction_content(prediction_files)
        await repository.mark_queued(request.submission_id, job_id)
    except FormatValidationError as exc:
        await _safe_mark_failed(repository, request.submission_id, str(exc))
        if not settings.retain_work_dir and job_dir.exists():
            shutil.rmtree(job_dir)
        return EvaluateResponse(accepted=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - request cannot be queued if persistence fails.
        logger.exception("Failed to queue evaluator job for submission %s", request.submission_id)
        await _safe_mark_failed(repository, request.submission_id, str(exc))
        if not settings.retain_work_dir and job_dir.exists():
            shutil.rmtree(job_dir)
        return EvaluateResponse(accepted=False, error="数据库更新失败，无法进入评分队列")

    background_tasks.add_task(
        run_scoring_job,
        submission_id=request.submission_id,
        topic_id=request.topic_id,
        prediction_dir=prediction_dir,
        repository=repository,
    )
    return EvaluateResponse(
        accepted=True,
        job_id=job_id,
        message="格式校验通过，已进入评分队列",
    )


def _validate_request_file_name(file_name: str) -> None:
    if "/" in file_name or "\\" in file_name:
        raise FormatValidationError("file_name 不能包含路径分隔符")
    if not file_name.lower().endswith(".zip"):
        raise FormatValidationError("提交文件必须是 zip 压缩包")


async def _safe_mark_failed(repository: SubmissionRepository, submission_id, error: str) -> None:
    try:
        await repository.mark_failed(submission_id, error)
    except Exception:  # noqa: BLE001 - preserve the original API response.
        logger.exception("Failed to persist validation error for submission %s", submission_id)
