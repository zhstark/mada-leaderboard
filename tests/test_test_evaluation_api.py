import asyncio
from uuid import UUID

from fastapi import BackgroundTasks

import evaluator_service.main as main_module
from evaluator_service.models import EvaluateRequest


class FakeTestSubmissionRepository:
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


def test_test_submission_queues_job_and_schedules_test_scoring(monkeypatch, tmp_path):
    prediction_dir = tmp_path / "predictions"
    ground_truth_dir = tmp_path / "q1_test" / "labels"
    prediction_dir.mkdir()
    ground_truth_dir.mkdir(parents=True)
    captured_repository = None
    scheduled_jobs = []

    async def download_zip(download_url, target_path, expected_size, settings):
        return None

    def extract_zip_safely(zip_path, destination, settings):
        return prediction_dir

    def validate_prediction_files(prediction_dir_arg, ground_truth_dir_arg):
        assert prediction_dir_arg == prediction_dir
        assert ground_truth_dir_arg == ground_truth_dir
        return [prediction_dir / "image-1.txt"]

    def validate_prediction_content(prediction_files):
        assert prediction_files == [prediction_dir / "image-1.txt"]

    def repository_factory(database_url):
        nonlocal captured_repository
        captured_repository = FakeTestSubmissionRepository(database_url)
        return captured_repository

    async def run_test_scoring_job(**kwargs):
        scheduled_jobs.append(kwargs)

    monkeypatch.setenv("EVALUATOR_WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("EVALUATOR_RETAIN_WORK_DIR", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(main_module, "download_zip", download_zip)
    monkeypatch.setattr(main_module, "extract_zip_safely", extract_zip_safely)
    monkeypatch.setattr(main_module, "validate_prediction_files", validate_prediction_files)
    monkeypatch.setattr(main_module, "validate_prediction_content", validate_prediction_content)
    monkeypatch.setattr(main_module, "test_ground_truth_dir_for_topic", lambda topic_id: ground_truth_dir)
    monkeypatch.setattr(main_module, "TestSubmissionRepository", repository_factory)
    monkeypatch.setattr(main_module, "run_test_scoring_job", run_test_scoring_job)

    submission_id = "33333333-3333-3333-3333-333333333333"
    background_tasks = BackgroundTasks()
    response = asyncio.run(
        main_module.create_test_submission(
            EvaluateRequest(
                submission_id=submission_id,
                topic_id=1,
                file_name="prediction.zip",
                file_size=123,
                download_url="https://example.com/prediction.zip",
            ),
            background_tasks,
        )
    )
    asyncio.run(background_tasks())

    assert response.accepted is True
    assert response.job_id is not None
    assert response.message == "格式校验通过，已进入评分队列"
    assert captured_repository.queued_submission_id == UUID(submission_id)
    assert captured_repository.queued_job_id == response.job_id
    assert len(scheduled_jobs) == 1
    assert scheduled_jobs[0]["submission_id"] == UUID(submission_id)
    assert scheduled_jobs[0]["topic_id"] == 1
    assert scheduled_jobs[0]["prediction_dir"] == prediction_dir
    assert scheduled_jobs[0]["repository"] is captured_repository


def test_q3_test_submission_accepts_single_xlsx_without_extracting(monkeypatch, tmp_path):
    prediction_path = tmp_path / "jobs" / "job-id" / "prediction.xlsx"
    ground_truth_path = tmp_path / "q3_test" / "test_result.xlsx"
    captured_repository = None
    scheduled_jobs = []

    async def download_xlsx(download_url, target_path, expected_size, settings):
        assert target_path.name == "prediction.xlsx"

    def validate_quality_submission(prediction_path_arg, ground_truth_path_arg):
        assert prediction_path_arg == prediction_path
        assert ground_truth_path_arg == ground_truth_path

    def repository_factory(database_url):
        nonlocal captured_repository
        captured_repository = FakeTestSubmissionRepository(database_url)
        return captured_repository

    async def run_test_scoring_job(**kwargs):
        scheduled_jobs.append(kwargs)

    monkeypatch.setattr("evaluator_service.main.uuid4", lambda: "job-id")
    monkeypatch.setenv("EVALUATOR_WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("EVALUATOR_RETAIN_WORK_DIR", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(main_module, "download_xlsx", download_xlsx)
    monkeypatch.setattr(main_module, "validate_quality_submission", validate_quality_submission)
    monkeypatch.setattr(main_module, "test_ground_truth_dir_for_topic", lambda topic_id: ground_truth_path)
    monkeypatch.setattr(main_module, "TestSubmissionRepository", repository_factory)
    monkeypatch.setattr(main_module, "run_test_scoring_job", run_test_scoring_job)

    submission_id = "55555555-5555-5555-5555-555555555555"
    background_tasks = BackgroundTasks()
    response = asyncio.run(
        main_module.create_test_submission(
            EvaluateRequest(
                submission_id=submission_id,
                topic_id=3,
                file_name="prediction.xlsx",
                file_size=123,
                download_url="https://example.com/prediction.xlsx",
            ),
            background_tasks,
        )
    )
    asyncio.run(background_tasks())

    assert response.accepted is True
    assert scheduled_jobs[0]["topic_id"] == 3
    assert scheduled_jobs[0]["prediction_dir"] == prediction_path
    assert scheduled_jobs[0]["repository"] is captured_repository


def test_q3_rejects_zip_file_name(monkeypatch, tmp_path):
    monkeypatch.setenv("EVALUATOR_WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("EVALUATOR_RETAIN_WORK_DIR", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(main_module, "TestSubmissionRepository", FakeTestSubmissionRepository)

    response = asyncio.run(
        main_module.create_test_submission(
            EvaluateRequest(
                submission_id="66666666-6666-6666-6666-666666666666",
                topic_id=3,
                file_name="prediction.zip",
                file_size=123,
                download_url="https://example.com/prediction.zip",
            ),
            BackgroundTasks(),
        )
    )

    assert response.accepted is False
    assert response.error == "q3 提交文件必须是 xlsx 文件"


def test_q4_test_submission_accepts_single_txt(monkeypatch, tmp_path):
    prediction_path = tmp_path / "jobs" / "job-id" / "prediction.txt"
    ground_truth_path = tmp_path / "q4_test" / "test.txt"
    scheduled_jobs = []

    async def download_txt(download_url, target_path, expected_size, settings):
        assert target_path == prediction_path

    def validate_count_submission(prediction_path_arg, ground_truth_path_arg):
        assert prediction_path_arg == prediction_path
        assert ground_truth_path_arg == ground_truth_path

    async def run_test_scoring_job(**kwargs):
        scheduled_jobs.append(kwargs)

    monkeypatch.setattr(main_module, "uuid4", lambda: "job-id")
    monkeypatch.setenv("EVALUATOR_WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("EVALUATOR_RETAIN_WORK_DIR", "1")
    monkeypatch.setattr(main_module, "download_txt", download_txt)
    monkeypatch.setattr(main_module, "validate_count_submission", validate_count_submission)
    monkeypatch.setattr(main_module, "test_ground_truth_dir_for_topic", lambda topic_id: ground_truth_path)
    monkeypatch.setattr(main_module, "TestSubmissionRepository", FakeTestSubmissionRepository)
    monkeypatch.setattr(main_module, "run_test_scoring_job", run_test_scoring_job)

    background_tasks = BackgroundTasks()
    response = asyncio.run(
        main_module.create_test_submission(
            EvaluateRequest(
                submission_id="88888888-8888-8888-8888-888888888888",
                topic_id=4,
                file_name="prediction.txt",
                file_size=123,
                download_url="https://example.com/prediction.txt",
            ),
            background_tasks,
        )
    )
    asyncio.run(background_tasks())

    assert response.accepted is True
    assert scheduled_jobs[0]["topic_id"] == 4
    assert scheduled_jobs[0]["prediction_dir"] == prediction_path
