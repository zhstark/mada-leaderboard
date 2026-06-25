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

    _log_submission_event(
        "Received evaluator submission",
        request,
        job_id,
        job_dir=job_dir,
        zip_path=zip_path,
    )

    try:
        _log_submission_event("Validating evaluator submission file name", request, job_id)
        _validate_request_file_name(request.file_name)
        ground_truth_dir = ground_truth_dir_for_topic(request.topic_id)
        _log_submission_event("Resolved evaluator ground truth", request, job_id, ground_truth_dir=ground_truth_dir)
        _log_submission_event("Downloading evaluator prediction zip", request, job_id, zip_path=zip_path)
        await download_zip(str(request.download_url), zip_path, request.file_size, settings)
        _log_submission_event("Downloaded evaluator prediction zip", request, job_id, zip_path=zip_path)
        _log_submission_event(
            "Extracting evaluator prediction zip",
            request,
            job_id,
            zip_path=zip_path,
            extracted_dir=extracted_dir,
        )
        prediction_dir = extract_zip_safely(zip_path, extracted_dir, settings)
        _log_submission_event("Extracted evaluator prediction zip", request, job_id, prediction_dir=prediction_dir)
        _log_submission_event(
            "Validating evaluator prediction file set",
            request,
            job_id,
            prediction_dir=prediction_dir,
            ground_truth_dir=ground_truth_dir,
        )
        prediction_files = validate_prediction_files(prediction_dir, ground_truth_dir)
        _log_submission_event(
            "Validated evaluator prediction file set",
            request,
            job_id,
            prediction_dir=prediction_dir,
            prediction_file_count=len(prediction_files),
        )
        _log_submission_event(
            "Validating evaluator prediction content",
            request,
            job_id,
            prediction_file_count=len(prediction_files),
        )
        validate_prediction_content(prediction_files)
        _log_submission_event(
            "Validated evaluator prediction content",
            request,
            job_id,
            prediction_file_count=len(prediction_files),
        )
        _log_submission_event("Marking evaluator submission queued", request, job_id)
        await repository.mark_queued(request.submission_id, job_id)
        _log_submission_event("Queued evaluator submission", request, job_id)
    except FormatValidationError as exc:
        logger.warning(
            "Rejected evaluator submission submission_id=%s topic_id=%s file_name=%s file_size=%s "
            "download_url=%s job_id=%s job_dir=%s zip_path=%s error=%s",
            request.submission_id,
            request.topic_id,
            request.file_name,
            request.file_size,
            request.download_url,
            job_id,
            job_dir,
            zip_path,
            exc,
        )
        await _safe_mark_failed(repository, request.submission_id, str(exc))
        if not settings.retain_work_dir and job_dir.exists():
            shutil.rmtree(job_dir)
        return EvaluateResponse(accepted=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - request cannot be queued if persistence fails.
        logger.exception(
            "Failed to queue evaluator job submission_id=%s topic_id=%s file_name=%s file_size=%s "
            "download_url=%s job_id=%s job_dir=%s zip_path=%s",
            request.submission_id,
            request.topic_id,
            request.file_name,
            request.file_size,
            request.download_url,
            job_id,
            job_dir,
            zip_path,
        )
        await _safe_mark_failed(repository, request.submission_id, str(exc))
        if not settings.retain_work_dir and job_dir.exists():
            shutil.rmtree(job_dir)
        return EvaluateResponse(accepted=False, error="数据库更新失败，无法进入评分队列")

    _log_submission_event("Scheduling evaluator scoring job", request, job_id, prediction_dir=prediction_dir)
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
        logger.exception(
            "Failed to persist validation error submission_id=%s validation_error=%s",
            submission_id,
            error,
        )


def _log_submission_event(message: str, request: EvaluateRequest, job_id: str, **context: object) -> None:
    log_context = {
        "submission_id": request.submission_id,
        "topic_id": request.topic_id,
        "file_name": request.file_name,
        "file_size": request.file_size,
        "download_url": request.download_url,
        "job_id": job_id,
        **context,
    }
    logger.info("%s %s", message, _format_log_context(log_context))


def _format_log_context(context: dict[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in context.items())
