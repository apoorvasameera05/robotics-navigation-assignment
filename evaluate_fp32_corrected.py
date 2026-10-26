from ultralytics import YOLO

model = YOLO(
    r"runs\detect\runs\detect\traffic_cone_quick\weights\best_openvino_model"
)

metrics = model.val(
    data="data/traffic_cone_public/data_local.yaml",
    split="test",
    imgsz=416,
    batch=1,
    device="cpu",
    workers=0,
    project="runs/detect",
    name="fp32_corrected_test",
    exist_ok=True
)

print("\nCORRECTED FP32 TEST RESULTS")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)