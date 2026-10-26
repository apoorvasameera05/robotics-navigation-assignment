"""Detect navigation objects and estimate approximate distances.

COCO baseline:
    python detect.py --source assets/demo.png --output output/annotated.jpg

Open-vocabulary detection:
    python detect.py --source assets/demo.png --weights yolov8s-worldv2.pt --open-vocabulary --output output/world_annotated.jpg

Video file:
    python detect.py --source input.mp4 --weights yolov8s-worldv2.pt --open-vocabulary --output output/annotated.mp4

Webcam:
    python detect.py --source 0 --weights yolov8s-worldv2.pt --open-vocabulary --output output/webcam.mp4
"""

import argparse
import os

import cv2
import yaml

from src.detector import load_detector, normalize_name
from src.distance import estimate_distance_m


def annotate_frame(
    frame,
    result,
    names,
    allowed,
    heights,
    camera,
    conf_threshold=0.20,
):
    """Annotate navigation objects and return their estimated distances."""

    annotated = frame.copy()
    image_width = frame.shape[1]
    detections = []

    for box in result.boxes:
        confidence = float(box.conf.item())

        # Ignore predictions below the configured confidence threshold.
        if confidence < conf_threshold:
            continue

        class_id = int(box.cls.item())

        if isinstance(names, (list, tuple)):
            raw_name = names[class_id]
        else:
            raw_name = names.get(class_id, str(class_id))

        name = normalize_name(raw_name)

        # Only retain the navigation categories configured by the user.
        if name not in allowed:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        observed_height = max(1, y2 - y1)

        real_height = heights.get(name)

        distance = None
        if real_height is not None:
            distance = estimate_distance_m(
                observed_height,
                real_height,
                image_width,
                camera["horizontal_fov_degrees"],
                camera.get("focal_length_px"),
            )

        if distance is not None:
            label = f"{name}: {distance:.1f} m ({confidence:.2f})"
        else:
            label = f"{name} ({confidence:.2f})"

        # Draw the bounding box and label.
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            2,
        )

        text_y = max(22, y1 - 8)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "class": name,
                "confidence": round(confidence, 3),
                "distance_m": (
                    round(float(distance), 2)
                    if distance is not None
                    else None
                ),
                "bounding_box": [x1, y1, x2, y2],
            }
        )

    return annotated, detections


def load_config(config_path):
    """Load and validate the project configuration."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    camera = config.get("camera")
    object_heights = config.get("object_heights_m")
    navigation_classes = config.get("navigation_classes")

    if not isinstance(camera, dict):
        raise ValueError("Missing or invalid 'camera' configuration.")

    if "horizontal_fov_degrees" not in camera:
        raise ValueError(
            "camera.horizontal_fov_degrees must be specified in config.yaml."
        )

    if not isinstance(object_heights, dict):
        raise ValueError("Missing or invalid 'object_heights_m' configuration.")

    if not isinstance(navigation_classes, list):
        raise ValueError("Missing or invalid 'navigation_classes' configuration.")

    fov = float(camera["horizontal_fov_degrees"])
    if not 0 < fov < 180:
        raise ValueError(
            "camera.horizontal_fov_degrees must be between 0 and 180."
        )

    heights = {
        normalize_name(name): float(height)
        for name, height in object_heights.items()
    }

    for name, height in heights.items():
        if height <= 0:
            raise ValueError(
                f"Object height for '{name}' must be greater than zero."
            )

    allowed = {
        normalize_name(name)
        for name in navigation_classes
    }

    return camera, heights, allowed


def process_image(model, source, output_path, camera, heights, allowed, args):
    """Detect and annotate objects in a single image."""

    image = cv2.imread(source)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    result = model.predict(
        image,
        conf=args.conf,
        iou=0.45,
        max_det=50,
        device=args.device,
        verbose=False,
    )[0]

    annotated, detections = annotate_frame(
        image,
        result,
        model.names,
        allowed,
        heights,
        camera,
        args.conf,
    )

    if not cv2.imwrite(output_path, annotated):
        raise RuntimeError(f"Could not save annotated image: {output_path}")

    print(f"Saved annotated image: {output_path}")
    print(f"Target detections: {detections if detections else 'none detected'}")
    print(
        "Distance estimates are approximate. "
        "Camera calibration and actual object dimensions affect accuracy."
    )

    if not args.open_vocabulary:
        print(
            "Note: Standard COCO weights do not have dedicated traffic-cone "
            "or road-barrier classes. Use YOLO-World or custom-trained weights "
            "for these objects."
        )


def process_video(model, source, output_path, camera, heights, allowed, args):
    """Detect and annotate objects in a video or webcam stream."""

    source_is_webcam = source.isdigit()
    video_source = int(source) if source_is_webcam else source

    capture = cv2.VideoCapture(video_source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video or webcam: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not determine video frame dimensions.")

    if not output_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        output_path = os.path.splitext(output_path)[0] + ".mp4"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_count = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            result = model.predict(
                frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]

            annotated, _ = annotate_frame(
                frame,
                result,
                model.names,
                allowed,
                heights,
                camera,
                args.conf,
            )

            writer.write(annotated)
            frame_count += 1

            cv2.imshow("Navigation detection - press q to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved annotated video: {output_path}")
    print(f"Frames processed: {frame_count}")


def main():
    """Parse arguments and run the detection pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Navigation object detection and approximate distance estimation."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image/video file or webcam index, e.g. 0.",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Model weights path or supported model name.",
    )
    parser.add_argument(
        "--open-vocabulary",
        action="store_true",
        help="Configure YOLO-World prompts for navigation objects.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the project YAML configuration.",
    )
    parser.add_argument(
        "--output",
        default="output/annotated.jpg",
        help="Path for the annotated output.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Minimum detection confidence, between 0 and 1.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu or 0 for the first CUDA GPU.",
    )

    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1.")

    camera, heights, allowed = load_config(args.config)

    if not allowed:
        parser.error("navigation_classes must contain at least one class.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = load_detector(
        args.weights,
        open_vocabulary=args.open_vocabulary,
    )

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    is_webcam = args.source.isdigit()
    is_video = is_webcam or args.source.lower().endswith(video_extensions)

    if is_video:
        process_video(
            model,
            args.source,
            args.output,
            camera,
            heights,
            allowed,
            args,
        )
    else:
        process_image(
            model,
            args.source,
            args.output,
            camera,
            heights,
            allowed,
            args,
        )


if __name__ == "__main__":
    main()