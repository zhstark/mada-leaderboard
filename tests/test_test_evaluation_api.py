from uuid import UUID

from fastapi.testclient import TestClient

import evaluator_service.main as main_module


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
    response = TestClient(main_module.app).post(
        "/test-submissions",
        json={
            "submission_id": submission_id,
            "topic_id": 1,
            "file_name": "prediction.zip",
            "file_size": 123,
            "download_url": "https://example.com/prediction.zip",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["job_id"] is not None
    assert body["message"] == "格式校验通过，已进入评分队列"
    assert captured_repository.queued_submission_id == UUID(submission_id)
    assert captured_repository.queued_job_id == body["job_id"]
    assert len(scheduled_jobs) == 1
    assert scheduled_jobs[0]["submission_id"] == UUID(submission_id)
    assert scheduled_jobs[0]["topic_id"] == 1
    assert scheduled_jobs[0]["prediction_dir"] == prediction_dir
    assert scheduled_jobs[0]["repository"] is captured_repository
