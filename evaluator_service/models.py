from __future__ import annotations

from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field


class EvaluateRequest(BaseModel):
    submission_id: UUID
    topic_id: int = Field(..., ge=1, le=2)
    file_name: str = Field(..., min_length=1)
    file_size: int = Field(..., gt=0)
    download_url: AnyHttpUrl


class EvaluateResponse(BaseModel):
    accepted: bool
    job_id: str | None = None
    message: str | None = None
    error: str | None = None
