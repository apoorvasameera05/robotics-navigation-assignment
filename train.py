"""Fine-tune YOLO on a custom dataset in Ultralytics YOLO format."""
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project="runs/navigation",
                name="train")
    print("Training finished. Check runs/navigation/train/weights/best.pt")

if __name__ == "__main__":
    main()
