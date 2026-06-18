from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .errors import FormatValidationError


@dataclass(frozen=True)
class Box:
    image_id: str
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (
            self.x_center - half_w,
            self.y_center - half_h,
            self.x_center + half_w,
            self.y_center + half_h,
        )


def collect_label_files(labels_dir: Path) -> dict[str, Path]:
    if not labels_dir.exists():
        raise FormatValidationError(f"标签目录不存在: {labels_dir}")
    if not labels_dir.is_dir():
        raise FormatValidationError(f"标签路径不是目录: {labels_dir}")

    label_files: dict[str, Path] = {}
    for path in labels_dir.rglob("*.txt"):
        if not path.is_file():
            continue
        image_id = path.stem
        if image_id in label_files:
            raise FormatValidationError(f"存在重复 label 文件名: {path.name}")
        label_files[image_id] = path
    return label_files


def validate_prediction_files(prediction_dir: Path, ground_truth_dir: Path) -> dict[str, Path]:
    expected = collect_label_files(ground_truth_dir)
    submitted = collect_label_files(prediction_dir)

    expected_ids = set(expected)
    submitted_ids = set(submitted)
    missing = sorted(expected_ids - submitted_ids)
    extra = sorted(submitted_ids - expected_ids)
    if missing:
        preview = ", ".join(f"{item}.txt" for item in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise FormatValidationError(f"提交缺少 {len(missing)} 个 label 文件: {preview}{suffix}")
    if extra:
        preview = ", ".join(f"{item}.txt" for item in extra[:5])
        suffix = " ..." if len(extra) > 5 else ""
        raise FormatValidationError(f"提交包含 {len(extra)} 个未知 label 文件: {preview}{suffix}")
    return submitted


def load_ground_truth(labels_dir: Path) -> list[Box]:
    boxes: list[Box] = []
    for image_id, path in collect_label_files(labels_dir).items():
        boxes.extend(_parse_label_file(path, image_id, expect_prediction=False))
    return boxes


def load_predictions(prediction_files: dict[str, Path]) -> list[Box]:
    boxes: list[Box] = []
    for image_id, path in prediction_files.items():
        boxes.extend(_parse_label_file(path, image_id, expect_prediction=True))
    return boxes


def validate_prediction_content(prediction_files: dict[str, Path]) -> None:
    for image_id, path in prediction_files.items():
        _parse_label_file(path, image_id, expect_prediction=True)


def _parse_label_file(path: Path, image_id: str, *, expect_prediction: bool) -> list[Box]:
    boxes: list[Box] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise FormatValidationError(f"{path.name} 不是 UTF-8 文本文件") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        expected_lengths = {5, 6} if expect_prediction else {5}
        if len(parts) not in expected_lengths:
            expected = "5 或 6" if expect_prediction else "5"
            raise FormatValidationError(f"{path.name}:{line_number} 应为 {expected} 列，实际为 {len(parts)} 列")

        try:
            class_id_float = float(parts[0])
            class_id = int(class_id_float)
            values = [float(item) for item in parts[1:]]
        except ValueError as exc:
            raise FormatValidationError(f"{path.name}:{line_number} 包含非数字字段") from exc

        if class_id_float != class_id or class_id < 0:
            raise FormatValidationError(f"{path.name}:{line_number} class_id 必须是非负整数")

        x_center, y_center, width, height = values[:4]
        confidence = values[4] if len(values) == 5 else 1.0
        _validate_box_numbers(path.name, line_number, x_center, y_center, width, height, confidence)
        boxes.append(
            Box(
                image_id=image_id,
                class_id=class_id,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                confidence=confidence,
            )
        )
    return boxes


def _validate_box_numbers(
    file_name: str,
    line_number: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    confidence: float,
) -> None:
    values = {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
        "confidence": confidence,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise FormatValidationError(f"{file_name}:{line_number} {name} 不是有限数值")

    if not 0.0 <= x_center <= 1.0 or not 0.0 <= y_center <= 1.0:
        raise FormatValidationError(f"{file_name}:{line_number} x_center/y_center 必须在 0 到 1 之间")
    if not 0.0 < width <= 1.0 or not 0.0 < height <= 1.0:
        raise FormatValidationError(f"{file_name}:{line_number} width/height 必须在 0 到 1 之间且大于 0")
    if not 0.0 <= confidence <= 1.0:
        raise FormatValidationError(f"{file_name}:{line_number} confidence 必须在 0 到 1 之间")
