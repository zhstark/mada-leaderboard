from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import httpx

from .config import Settings
from .errors import FormatValidationError


async def download_zip(download_url: str, target_path: Path, expected_size: int, settings: Settings) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(settings.download_timeout_seconds)
    bytes_written = 0

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            async with client.stream("GET", download_url) as response:
                if response.status_code >= 400:
                    raise FormatValidationError(f"下载预测文件失败，HTTP {response.status_code}")
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > settings.max_download_bytes:
                    raise FormatValidationError("预测 zip 文件超过服务允许的最大大小")

                with target_path.open("wb") as output:
                    async for chunk in response.aiter_bytes():
                        bytes_written += len(chunk)
                        if bytes_written > settings.max_download_bytes:
                            raise FormatValidationError("预测 zip 文件超过服务允许的最大大小")
                        output.write(chunk)
        except httpx.HTTPError as exc:
            raise FormatValidationError(f"下载预测文件失败: {exc}") from exc

    if bytes_written != expected_size:
        raise FormatValidationError(f"下载文件大小不一致: 期望 {expected_size} 字节，实际 {bytes_written} 字节")
    if not zipfile.is_zipfile(target_path):
        raise FormatValidationError("提交文件不是有效的 zip 压缩包")


def extract_zip_safely(zip_path: Path, destination: Path, settings: Settings) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    total_uncompressed_size = 0

    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            if len(members) > settings.max_zip_members:
                raise FormatValidationError("zip 内文件数量超过服务允许的最大数量")

            for member in members:
                if member.is_dir():
                    continue
                total_uncompressed_size += member.file_size
                if total_uncompressed_size > settings.max_uncompressed_bytes:
                    raise FormatValidationError("zip 解压后体积超过服务允许的最大大小")
                target_path = (destination / member.filename).resolve()
                if destination_root not in target_path.parents:
                    raise FormatValidationError("zip 文件包含非法路径")

            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise FormatValidationError("提交文件不是有效的 zip 压缩包") from exc

    return _find_label_root(destination)


def _find_label_root(extracted_dir: Path) -> Path:
    candidates = [
        extracted_dir / "labels",
        extracted_dir / "labels" / "val",
    ]
    for child in extracted_dir.iterdir():
        if child.is_dir():
            candidates.extend([child / "labels", child / "labels" / "val"])

    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.txt")):
            return candidate
    return extracted_dir
