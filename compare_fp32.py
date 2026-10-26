from ultralytics import YOLO

model = YOLO("yolov8n_openvino_model")

metrics = model.val(
    data="data/traffic_cone_public/data_local.yaml",
    split="test",
    imgsz=416,
    batch=8,
    device="cpu",
    workers=2,
    project="runs/detect",
    name="fp32_test",
    exist_ok=True
)

print("\nFP32 TEST RESULTS")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
