from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .errors import FormatValidationError


@dataclass(frozen=True)
class CountMeasurement:
    image_id: str
    count: float


def load_count_file(path: Path) -> dict[str, CountMeasurement]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FormatValidationError(f"{path.name} 不是有效的 UTF-8 文本文件") from exc

    measurements: dict[str, CountMeasurement] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise FormatValidationError(f"{path.name}:{line_number} 应为 2 列，实际为 {len(parts)} 列")

        image_id, raw_count = parts
        if image_id in measurements:
            raise FormatValidationError(f"{path.name}:{line_number} 图片文件名重复: {image_id}")
        try:
            count = float(raw_count)
        except ValueError as exc:
            raise FormatValidationError(f"{path.name}:{line_number} 数量必须是数值") from exc
        if not math.isfinite(count):
            raise FormatValidationError(f"{path.name}:{line_number} 数量必须是有限数值")
        if count < 0:
            raise FormatValidationError(f"{path.name}:{line_number} 数量不能为负数")
        measurements[image_id] = CountMeasurement(image_id, count)

    if not measurements:
        raise FormatValidationError(f"{path.name} 不包含有效数据行")
    return measurements


def validate_count_submission(prediction_path: Path, ground_truth_path: Path) -> None:
    ground_truth = load_count_file(ground_truth_path)
    predictions = load_count_file(prediction_path)
    _validate_image_ids(ground_truth, predictions, prediction_path.name)


def load_count_evaluation_data(
    prediction_path: Path,
    ground_truth_path: Path,
) -> tuple[dict[str, CountMeasurement], dict[str, CountMeasurement]]:
    ground_truth = load_count_file(ground_truth_path)
    predictions = load_count_file(prediction_path)
    _validate_image_ids(ground_truth, predictions, prediction_path.name)
    return ground_truth, predictions


def _validate_image_ids(
    ground_truth: dict[str, CountMeasurement],
    predictions: dict[str, CountMeasurement],
    prediction_name: str,
) -> None:
    expected_ids = set(ground_truth)
    submitted_ids = set(predictions)
    missing = sorted(expected_ids - submitted_ids)
    extra = sorted(submitted_ids - expected_ids)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FormatValidationError(f"{prediction_name} 缺少 {len(missing)} 个图片: {preview}{suffix}")
    if extra:
        preview = ", ".join(extra[:5])
        suffix = " ..." if len(extra) > 5 else ""
        raise FormatValidationError(f"{prediction_name} 包含 {len(extra)} 个未知图片: {preview}{suffix}")
