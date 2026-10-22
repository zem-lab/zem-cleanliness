"""Task-local SEM image sampling settings; physical field of view is unchanged."""
from dataclasses import replace
import math


def validate_capture_image_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value not in (512, 1024):
        raise ValueError("captureImageSize 必须是 512 或 1024")
    return value


def calibration_with_image_size(calibration, task_settings):
    size = task_settings.get("captureImageSize")
    if size is None:
        return calibration  # Historical task snapshots retain their imaging profile.
    size = validate_capture_image_size(size)
    imaging = dict(calibration.imaging)
    field = float(imaging.get("fieldSizeUm") or 0)
    if not math.isfinite(field) or field <= 0:
        raise ValueError("无法设置图幅：标定文件缺少有效物理视场")
    imaging.update(imageShape=[size, size], pixelSizeUm=field / size)
    return replace(calibration, imaging=imaging)
