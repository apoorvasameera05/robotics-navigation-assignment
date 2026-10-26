"""Measure YOLO inference latency and throughput on a repeated image."""

import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO inference latency and throughput."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="cpu", help="cpu or 0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--open-vocabulary", action="store_true")
    args = parser.parse_args()

    if args.runs <= 0 or args.warmup < 0 or args.imgsz <= 0:
        parser.error("--runs and --imgsz must be positive; --warmup cannot be negative")

    image = cv2.imread(args.source)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.source}")

    print(f"Loading model: {args.weights}", flush=True)
    model = YOLO(args.weights)

    if args.open_vocabulary:
        model.set_classes(["traffic cone", "road barrier", "stop sign"])

    print(f"Image size: {args.imgsz}", flush=True)
    print(f"Warm-up: {args.warmup} runs", flush=True)

    for _ in range(args.warmup):
        model.predict(
            image,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

    print(f"Benchmarking: {args.runs} runs", flush=True)

    start = time.perf_counter()

    for _ in range(args.runs):
        model.predict(
            image,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

    elapsed = time.perf_counter() - start
    latency_ms = elapsed / args.runs * 1000
    fps = args.runs / elapsed

    print("\n--- Benchmark Results ---")
    print(f"Model: {args.weights}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.imgsz}")
    print(f"Runs: {args.runs}")
    print(f"Total time: {elapsed:.3f} s")
    print(f"Average latency: {latency_ms:.2f} ms/image")
    print(f"Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    main()