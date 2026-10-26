"""Export YOLO weights to ONNX; ONNX export does not itself guarantee faster inference."""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    model = YOLO(args.weights)
    output = model.export(format="onnx", imgsz=args.imgsz)
    print(f"Exported model: {output}")
    print("Benchmark the exported model before claiming speed or accuracy improvements.")

if __name__ == "__main__":
    main()
