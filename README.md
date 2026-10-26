# Object Detection and Distance Estimation for Robotics Navigation

## Contact Information

- **Author:** Kodali Apoorva Sameera
- **Phone Number:** 9032217448
- **Email Address:** apoorvasameerakodali@gmail.com
- **GitHub:** https://github.com/apoorvasameera05
**GitHub Username:** [apoorvasameera05](https://github.com/apoorvasameera05)  
**GitHub Repository:** https://github.com/apoorvasameera05/robotics-navigation-assignment

## Overview

This project implements an image and video pipeline for detecting objects relevant to robotics navigation, drawing bounding boxes, estimating approximate distances using monocular camera geometry, and benchmarking inference performance across available inference devices.

It includes a COCO-pretrained YOLO baseline and an optional YOLO-World open-vocabulary detection mode for identifying traffic cones, road barriers, and stop signs without initially training a custom detector.

This project is a prototype intended for experimentation and learning. It is not a safety-certified robotics navigation system.

## Target Objects

- Traffic cones
- Road barriers and barricades
- Stop signs

## Features

- Image, video, and webcam input support
- Object bounding boxes, class labels, and confidence scores
- Approximate distance estimation in metres
- CPU benchmarking and preliminary Intel Iris Xe GPU benchmarking through OpenVINO
- ONNX model export experiment
- OpenVINO FP32 and INT8 export and benchmark experiments
- Custom YOLO-format fine-tuning experiment

## Project Setup — Windows PowerShell

### 1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Baseline Detector

Run the detector on the included demo image:

```powershell
python detect.py --source assets/demo.png --output output/annotated.jpg
```

The default `yolov8n.pt` model is pretrained on the COCO dataset. It can detect stop signs, but COCO does not include dedicated traffic-cone and road-barrier classes. Therefore, the baseline model is not expected to detect all three target objects reliably.

The annotated output is saved to the specified output path.

## Open-Vocabulary Detection with YOLO-World

YOLO-World supports text-prompt-based object detection and can be used to experiment with traffic cones, road barriers, and stop signs.

Run:

```powershell
python detect.py --source assets/demo.png --weights yolov8s-worldv2.pt --open-vocabulary --output output/world_annotated.jpg
```

The first run downloads the model weights and requires an internet connection. Open-vocabulary detections are not guaranteed. Inspect the annotated output to verify which objects were detected.

During testing, YOLO-World detected traffic cones in the demo image, but road-barrier detection remained unreliable. For more reliable results, collect and label a representative dataset and fine-tune a detector for the intended navigation environment.

## Distance Estimation

The project estimates distance using the apparent height of a detected object:

```text
distance_m = focal_length_px * assumed_real_height_m / bounding_box_height_px
```

Where:

- `distance_m` is the estimated distance in metres.
- `focal_length_px` is the camera focal length expressed in pixels.
- `assumed_real_height_m` is the assumed real-world height of the object.
- `bounding_box_height_px` is the detected object's height in pixels.

If a focal length is not supplied, the implementation estimates it using the image width and the configured horizontal field of view.

The representative object heights configured in `config.yaml` are:

| Object | Assumed height |
|---|---:|
| Traffic cone | 0.70 m |
| Road barrier | 1.00 m |
| Stop sign | 0.75 m |

These are assumptions, not universal dimensions. Actual object sizes can vary. Distance estimates may be inaccurate if the camera field of view, object dimensions, viewing angle, or detected bounding box is inaccurate.

Calibration against objects at known distances is necessary before practical use. These estimates must not be treated as reliable collision-avoidance measurements without further validation.

## Benchmarking

Run the baseline CPU benchmark using:

```powershell
python benchmark.py --source assets/demo.png --weights yolov8n.pt --device cpu --runs 50
```

The benchmark measures inference latency and throughput for the specified model and input image. Results depend on hardware, model configuration, and inference settings. The benchmark repeatedly processes the same image and measures Ultralytics prediction time. Actual performance may differ when processing live camera feeds or running on the target robotics hardware.

### CPU Performance Benchmark

The following measurements were recorded during development using the same demo image and CPU, with 50 inference runs per model.

| Metric | YOLOv8n, 640 px | YOLOv8n, 416 px | YOLOv8s-World, 640 px |
|---|---:|---:|---:|
| Device | CPU | CPU | CPU |
| Benchmark runs | 50 | 50 | 50 |
| Total benchmark time | 1.889 s | 1.106 s | 4.314 s |
| Average latency | 37.77 ms/image | 22.98 ms/image | 86.27 ms/image |
| Throughput | 26.47 FPS | 43.51 FPS | 11.59 FPS |

### Observations

- YOLOv8n achieved higher throughput and lower latency in this benchmark.
- YOLOv8s-World detected traffic cones in the demo image that the standard COCO model missed.
- YOLOv8s-World was approximately 2.25 times slower than YOLOv8n in this test.
- These measurements represent a single demo image processed on a development CPU.
- The results do not establish real-time performance on a live camera or the target robotics hardware.
- Road-barrier detection remains unreliable and requires further validation or fine-tuning on a labelled dataset.

Reducing YOLOv8n input size from 640 to 416 pixels reduced measured latency by approximately 39.2% and increased throughput by approximately 64.4% on the development CPU. Detection quality at the smaller resolution must also be evaluated.

## OpenVINO CPU and Intel Iris Xe GPU Benchmark

YOLOv8n was exported to OpenVINO format at an input size of 416 × 416 pixels. The following preliminary run used five warm-up inferences and 30 measured inferences per device.

| Metric | OpenVINO CPU | OpenVINO Intel Iris Xe GPU |
|---|---:|---:|
| Total time (30 measured runs) | 1.414 s | 1.270 s |
| Average latency | 47.12 ms/image | 42.33 ms/image |
| Throughput | 21.22 FPS | 23.63 FPS |

In this run, the Intel Iris Xe GPU achieved approximately 11.4% higher throughput and 10.2% lower average latency than the OpenVINO CPU configuration. These measurements represent repeated single-image inference, not end-to-end camera or robotics navigation performance. Results are preliminary and may vary with hardware load and software configuration.

## OpenVINO INT8 Quantization Experiment

An INT8 OpenVINO model was exported using a 416 × 416 input size. The original calibration dataset available for this experiment was small, so these measurements should be treated as preliminary.

The earlier baseline benchmark did not demonstrate a consistent speed improvement from INT8 quantization across the tested devices. The exported INT8 model directory was approximately 3.5 MB; this is not a direct same-format comparison with the 6.2 MB PyTorch checkpoint.

### Corrected Custom-Model Evaluation: FP32 vs INT8

The custom traffic-cone model was exported to OpenVINO FP32 and INT8 formats and evaluated on the same 1,000-image test split at 416 × 416 resolution.

**Hardware:** Intel 11th Gen Core i5-1145G7 CPU  
**Runtime:** OpenVINO  
**Evaluation batch size:** 1

| Metric | FP32 | INT8 |
|---|---:|---:|
| Precision | 53.69% | 53.00% |
| Recall | 23.75% | 23.60% |
| mAP@0.50 | 19.66% | 19.20% |
| mAP@0.50:0.95 | 8.82% | 8.61% |
| Inference latency | 9.2 ms/image | 6.4 ms/image |

In this evaluation, INT8 achieved approximately 30.4% lower reported inference latency than FP32. Precision, recall, and mAP decreased slightly after quantization.

These measurements indicate a potential speed-versus-accuracy trade-off on the tested hardware. Timing may vary with system load and runtime conditions.

The `unknown_cone` class remained a major weakness, with approximately zero recall in the INT8 evaluation. Overall detection performance is limited, and the model should not be considered ready for reliable real-world navigation.

The INT8 calibration experiment used 200 images. A larger and more representative calibration set is recommended for more reliable quantization evaluation.

## ONNX Export and Runtime

Export the baseline model to ONNX using:

```powershell
python export_onnx.py --weights yolov8n.pt
```

ONNX export succeeded in the development environment and produced `yolov8n.onnx`. An ONNX Runtime benchmark was attempted, but the Windows environment exited with status `-1073741819` (native access violation) before reporting measurements.

Consequently, no ONNX inference FPS or speedup is claimed. Exporting a model to ONNX does not guarantee faster inference. Further testing in a stable environment is needed before making performance claims.

## Fine-Tuning on a Custom Dataset

A one-epoch fine-tuning experiment was completed using a small subset of 200 traffic-cone training images and five cone-related classes. The experiment used a YOLOv8n pretrained model, a 320-pixel input size, and CPU training.

The same 200 images were used for training and validation in this quick experiment, so the validation results do not provide an independent measure of generalization. The fine-tuned model produced some high-confidence cone detections but also generated false positives and oversized bounding boxes.

A subsequent corrected evaluation of the exported custom model was performed on a 1,000-image test split. The results are included in the FP32-versus-INT8 comparison above.

The custom dataset is not included in this repository. Further work should include longer training, separate training and validation sets, evaluation on unseen images, and appropriate detection metrics.

## Edge Optimization Ideas

Potential improvements to investigate include:

1. Use `yolov8n.pt` as the lightweight baseline.
2. Test smaller input image sizes, such as 416 or 320 pixels, while keeping detection and benchmark configurations consistent.
3. Compare PyTorch, ONNX Runtime, and Intel OpenVINO using equivalent inputs and repeated runs.
4. Investigate quantization or pruning after establishing baseline accuracy and speed.
5. Record latency, FPS, model size, and detection quality together.
6. Validate detection performance on representative road scenes before considering robotics deployment.

A faster detector that misses important navigation objects is not necessarily a better solution.

## Repository Structure

```text
robotics-navigation-assignment/
├── assets/
├── output/
├── src/
│   ├── distance.py
│   └── detector.py
├── detect.py
├── benchmark.py
├── benchmark_openvino.py
├── compare_fp32.py
├── compare_int8.py
├── evaluate_fp32_corrected.py
├── evaluate_int8_corrected.py
├── export_onnx.py
├── train.py
├── config.yaml
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
├── REPORT.md
└── README.md
```

The datasets, virtual environment, downloaded checkpoints, training outputs, and exported model directories may be excluded from version control to avoid unnecessary repository size. Refer to the report for the experimental setup, results, and limitations.

## Limitations

- Evaluation is limited to demo images, benchmark runs, and a small custom-model experiment.
- The COCO-pretrained baseline does not have dedicated traffic-cone or road-barrier classes.
- YOLO-World detection of road barriers remains unreliable.
- The one-epoch fine-tuned model produced false positives and oversized bounding boxes.
- The quick fine-tuning experiment used the same images for training and validation, so it does not provide an independent generalization assessment.
- The custom model's corrected test metrics remain low.
- Distance estimates are approximate and depend on camera calibration, assumed object dimensions, and bounding-box quality.
- ONNX export succeeded, but ONNX Runtime benchmarking failed in the development Windows environment; no ONNX speedup is claimed.
- OpenVINO CPU and Intel Iris Xe GPU results are preliminary and hardware-specific.
- INT8 results show lower latency in the corrected custom-model comparison, but detection accuracy also decreased slightly.
- Real-world navigation reliability and safety have not been established.

## Future Work

1. Collect and label a representative dataset of traffic cones, road barriers, and stop signs.
2. Train for more epochs using separate training and validation datasets, then evaluate on unseen images.
3. Measure precision, recall, and detection quality at multiple input resolutions.
4. Calibrate the camera and validate distance estimates against objects at known physical distances.
5. Compare inference runtimes in a stable environment using consistent benchmark settings.
6. Use a larger representative calibration dataset for INT8 quantization and compare both speed and detection accuracy.
7. Test with live camera input and the intended robotics hardware.
8. Investigate pruning and other optimizations only while tracking detection quality as well as speed.

## Conclusion

The project demonstrates a modular object-detection and monocular distance-estimation pipeline. CPU benchmarking showed that reducing YOLOv8n input size from 640 to 416 pixels improved measured throughput on the development machine. YOLO-World provided an alternative for detecting navigation-related objects beyond the baseline model's dedicated classes. OpenVINO CPU and Intel Iris Xe GPU inference were also benchmarked, and an INT8 quantization experiment was conducted.

In the corrected custom-model evaluation, INT8 reduced reported inference latency from 9.2 ms/image to 6.4 ms/image, while mAP@0.50 decreased from 19.66% to 19.20%. The fine-tuning experiment and corrected evaluation also exposed limitations in the current custom model. Further training, representative dataset evaluation, camera calibration, and target-hardware testing are necessary before practical robotics deployment.

## Contact Information

- **Full Name:** Kodali Apoorva Sameera
- **GitHub Username:** apoorvasameera05
- **GitHub Repository:** https://github.com/apoorvasameera05/robotics-navigation-assignment
- **Phone Number:** 9032217448
- **Email Address:** apoorvasameerakodali@gmail.com