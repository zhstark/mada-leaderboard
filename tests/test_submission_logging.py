import asyncio
import logging
from uuid import UUID

from fastapi import BackgroundTasks

import evaluator_service.main as main_module
from evaluator_service.errors import FormatValidationError
from evaluator_service.models import EvaluateRequest


class FakeSubmissionRepository:
    def __init__(self, database_url):
        self.database_url = database_url
        self.queued_submission_id = None
        self.queued_job_id = None

    async def mark_failed(self, submission_id, error):
        self.failed_submission_id = submission_id
        self.failed_error = error

    async def mark_queued(self, submission_id, job_id):
        self.queued_submission_id = submission_id
        self.queued_job_id = job_id


def test_submission_validation_failure_logs_request_context(monkeypatch, tmp_path, caplog):
    async def fail_download(download_url, target_path, expected_size, settings):
        raise FormatValidationError("下载预测文件失败: boom")

    monkeypatch.setenv("EVALUATOR_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("EVALUATOR_RETAIN_WORK_DIR", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(main_module, "download_zip", fail_download)
    monkeypatch.setattr(main_module, "SubmissionRepository", FakeSubmissionRepository)

    submission_id = UUID("11111111-1111-1111-1111-111111111111")
    request = EvaluateRequest(
        submission_id=submission_id,
        topic_id=1,
        file_name="prediction.zip",
        file_size=12345,
        download_url="https://example.com/prediction.zip?token=abc",
    )

    caplog.set_level(logging.WARNING)

    response = asyncio.run(main_module.create_submission(request, BackgroundTasks()))

    assert response.accepted is False
    assert response.error == "下载预测文件失败: boom"
    assert "Rejected evaluator submission" in caplog.text
    assert f"submission_id={submission_id}" in caplog.text
    assert "topic_id=1" in caplog.text
    assert "file_name=prediction.zip" in caplog.text
    assert "file_size=12345" in caplog.text
    assert "download_url=https://example.com/prediction.zip?token=abc" in caplog.text
    assert "job_dir=" in caplog.text
    assert "zip_path=" in caplog.text
    assert "error=下载预测文件失败: boom" in caplog.text


def test_submission_validation_success_logs_each_process_stage(monkeypatch, tmp_path, caplog):
    async def download_zip(download_url, target_path, expected_size, settings):
        return None

    def extract_zip_safely(zip_path, destination, settings):
        return tmp_path / "extracted" / "labels"

    def validate_prediction_files(prediction_dir, ground_truth_dir):
        return [prediction_dir / "image-1.txt"]

    def validate_prediction_content(prediction_files):
        return None

    monkeypatch.setenv("EVALUATOR_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("EVALUATOR_RETAIN_WORK_DIR", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(main_module, "download_zip", download_zip)
    monkeypatch.setattr(main_module, "extract_zip_safely", extract_zip_safely)
    monkeypatch.setattr(main_module, "validate_prediction_files", validate_prediction_files)
    monkeypatch.setattr(main_module, "validate_prediction_content", validate_prediction_content)
    monkeypatch.setattr(main_module, "ground_truth_dir_for_topic", lambda topic_id: tmp_path / f"q{topic_id}")
    monkeypatch.setattr(main_module, "SubmissionRepository", FakeSubmissionRepository)

    submission_id = UUID("22222222-2222-2222-2222-222222222222")
    request = EvaluateRequest(
        submission_id=submission_id,
        topic_id=2,
        file_name="prediction.zip",
        file_size=54321,
        download_url="https://example.com/prediction.zip?token=def",
    )

    caplog.set_level(logging.INFO)

    response = asyncio.run(main_module.create_submission(request, BackgroundTasks()))

    assert response.accepted is True
    assert response.job_id is not None
    for message in [
        "Received evaluator submission",
        "Validating evaluator submission file name",
        "Resolved evaluator ground truth",
        "Downloading evaluator prediction zip",
        "Downloaded evaluator prediction zip",
        "Extracting evaluator prediction zip",
        "Extracted evaluator prediction zip",
        "Validating evaluator prediction file set",
        "Validated evaluator prediction file set",
        "Validating evaluator prediction content",
        "Validated evaluator prediction content",
        "Marking evaluator submission queued",
        "Queued evaluator submission",
        "Scheduling evaluator scoring job",
    ]:
        assert message in caplog.text
    assert f"submission_id={submission_id}" in caplog.text
    assert "topic_id=2" in caplog.text
    assert "file_name=prediction.zip" in caplog.text
    assert "file_size=54321" in caplog.text
    assert "download_url=https://example.com/prediction.zip?token=def" in caplog.text
