import time
from ultralytics import YOLO

model = YOLO("yolov8n_openvino_model")
source = "assets/demo.png"

for device in ["intel:gpu", "cpu"]:
    print("\nBenchmarking", device)

    for _ in range(5):
        model.predict(source=source, device=device, imgsz=416, verbose=False)

    start = time.perf_counter()

    for _ in range(30):
        model.predict(source=source, device=device, imgsz=416, verbose=False)

    elapsed = time.perf_counter() - start

    print("Total time:", round(elapsed, 3), "seconds")
    print("Average time:", round(elapsed / 30 * 1000, 2), "ms/image")
    print("Throughput:", round(30 / elapsed, 2), "FPS")
