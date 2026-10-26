"""Approximate monocular distance estimation using apparent object height."""

import math


def focal_length_from_fov(
    image_width_px: int,
    horizontal_fov_degrees: float,
) -> float:
    """Calculate camera focal length in pixels from horizontal FOV.

    Args:
        image_width_px: Width of the image in pixels.
        horizontal_fov_degrees: Camera's horizontal field of view in degrees.

    Returns:
        Focal length in pixels.

    Raises:
        ValueError: If image width or field of view is invalid.
    """
    if image_width_px <= 0:
        raise ValueError("Image width must be positive.")

    if not math.isfinite(horizontal_fov_degrees) or not (
        0 < horizontal_fov_degrees < 180
    ):
        raise ValueError(
            "Horizontal FOV must be between 0 and 180 degrees."
        )

    return image_width_px / (
        2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    )


def estimate_distance_m(
    object_height_px: int,
    real_object_height_m: float,
    image_width_px: int,
    horizontal_fov_degrees: float = 70.0,
    focal_length_px: float | None = None,
) -> float | None:
    """Estimate object distance in metres using its apparent height.

    Formula:
        D = (f * H) / h

    Args:
        object_height_px: Detected object's bounding-box height in pixels.
        real_object_height_m: Assumed real-world object height in metres.
        image_width_px: Input image width in pixels.
        horizontal_fov_degrees: Camera's horizontal field of view in degrees.
        focal_length_px: Optional calibrated focal length in pixels.

    Returns:
        Estimated distance in metres, or None if the object measurements
        are invalid.

    Note:
        The result is approximate. Accuracy depends on camera calibration,
        object dimensions, and the quality of the detected bounding box.
    """
    if (
        object_height_px <= 0
        or real_object_height_m <= 0
        or image_width_px <= 0
    ):
        return None

    if not math.isfinite(real_object_height_m):
        return None

    if focal_length_px is not None:
        if not math.isfinite(focal_length_px) or focal_length_px <= 0:
            return None

        f_px = focal_length_px
    else:
        f_px = focal_length_from_fov(
            image_width_px=image_width_px,
            horizontal_fov_degrees=horizontal_fov_degrees,
        )

    distance_m = (
        f_px * real_object_height_m
    ) / object_height_px

    return float(distance_m)
