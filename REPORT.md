# Object Detection and Distance Estimation for Robotics Navigation

## 1. Objective

The objective of this project is to develop a prototype computer-vision pipeline for robotics navigation. The system detects traffic cones, road barriers, and stop signs, estimates their approximate distances from monocular images, and measures inference performance on a CPU.

The project explores lightweight object detection, open-vocabulary detection, monocular distance estimation, inference benchmarking, and optimization techniques relevant to resource-constrained robotics applications.

The primary objectives are:

- Detect navigation-related objects, including traffic cones, road barriers, and stop signs.

- Estimate approximate object distances using bounding-box dimensions and camera geometry.

- Measure inference latency and throughput under different input resolutions.

- Explore alternative inference formats and hardware backends.

- Conduct a preliminary custom-dataset fine-tuning experiment.

- Document implementation limitations and identify future improvements.

## 2. System Architecture

The implementation is organized into modular components.

| Component | Purpose |

|---|---|

| `detect.py` | Runs object detection on images, videos, or webcam input and saves annotated results. |

| `src/detector.py` | Loads the detection model and normalizes detected object labels. |

| `src/distance.py` | Estimates object distance using bounding-box height and camera geometry. |

| `benchmark.py` | Measures inference latency and throughput. |

| `benchmark_openvino.py` | Benchmarks OpenVINO inference configurations. |

| `export_onnx.py` | Exports the baseline model to ONNX format. |

| `train.py` | Provides an entry point for training with a custom YOLO-format dataset. |

| `config.yaml` | Stores camera parameters, assumed object heights, and target classes. |

Two primary detection approaches are explored:

1. **YOLOv8n:** A lightweight model pretrained on the COCO dataset.

2. **YOLOv8s-World:** An open-vocabulary detection model that supports text prompts for identifying objects.

The project also explores OpenVINO inference and ONNX model export as potential deployment and optimization options.

## 3. Object Detection

### 3.1 YOLOv8n Baseline Model

YOLOv8n is used as the lightweight baseline detector.

Its small model size makes it suitable for investigating inference performance on CPU-based systems. However, its standard COCO training classes do not include dedicated traffic-cone and road-barrier categories.

Consequently, the baseline model is not expected to detect all the required navigation objects reliably without additional training or an alternative detection approach.

### 3.2 YOLOv8s-World Open-Vocabulary Detection

YOLOv8s-World supports text-prompt-based object detection. The prompts used in this project include:

- Traffic cone

- Roadside concrete barrier

- Stop sign

This approach provides flexibility when experimenting with objects that are not directly represented by dedicated classes in the baseline model.

### 3.3 Demonstration Results

The system was tested on a demonstration image, and an annotated output was generated at:

`output/final_demo.jpg`

The demonstration includes detected objects, bounding boxes, and approximate distance estimates.

During qualitative testing:

- Traffic cones were detected.

- A stop sign was detected.

- Road-barrier detection was less reliable.

These observations demonstrate the pipeline's functionality on the selected example. They are not a formal evaluation of detection accuracy.

A more comprehensive evaluation would require a representative test dataset with ground-truth annotations and quantitative metrics such as precision, recall, and mean Average Precision (mAP).

## 4. Monocular Distance Estimation

### 4.1 Distance Estimation Formula

The system estimates object distance using the relationship between an object's assumed physical height and its height in the image.

The formula is:

\[

d = \frac{f \times H}{h}

\]

Where:

- \(d\) is the estimated distance in metres.

- \(f\) is the camera focal length in pixels.

- \(H\) is the assumed physical height of the object in metres.

- \(h\) is the detected bounding-box height in pixels.

The calculation assumes that the object dimensions and camera parameters are known or appropriately estimated.

### 4.2 Estimating the Focal Length

If the focal length is not supplied, it is estimated using the image width and horizontal field of view:

\[

f = \frac{W}{2\tan(\theta/2)}

\]

Where:

- \(W\) is the image width in pixels.

- \(\theta\) is the horizontal field of view.

### 4.3 Configured Object Heights

The following representative physical heights are configured for distance estimation.

| Object category | Assumed physical height |

|---|---:|

| Traffic cone | 0.70 m |

| Road barrier | 1.00 m |

| Stop sign | 0.75 m |

These are assumptions rather than measurements of the objects in the demonstration image.

Actual object dimensions can vary, so these values may introduce estimation errors.

### 4.4 Numerical Verification

Consider the following example:

- Image width: 1280 pixels

- Horizontal field of view: 70°

- Assumed object height: 0.70 m

- Bounding-box height: 200 pixels

First, calculate the focal length:

\[

f = \frac{1280}{2\tan(70^\circ/2)}

\]

\[

f \approx 914.1 \text{ pixels}

\]

Then calculate the estimated distance:

\[

d = \frac{914.1 \times 0.70}{200}

\]

\[

d \approx 3.20 \text{ metres}

\]

The estimated distance is approximately **3.20 metres**.

This numerical test verifies the calculation, not the accuracy of the system in real-world conditions.

### 4.5 Limitations

Monocular distance estimation is sensitive to several factors:

- Incorrect assumptions about physical object dimensions.

- Inaccurate camera focal length or field-of-view parameters.

- Errors in detected bounding-box dimensions.

- Perspective effects and object orientation.

- Camera calibration errors.

Validation against objects placed at known physical distances is required to quantify real-world distance-estimation accuracy.

## 5. CPU Benchmarking

### 5.1 Experimental Setup

The baseline model was benchmarked on the development machine using CPU inference.

The experiments used:

- Five warm-up runs.

- Fifty measured inference runs per configuration.

- A repeated demonstration image.

- Input resolutions of 640 × 640 and 416 × 416 pixels for YOLOv8n.

The YOLOv8s-World model was also benchmarked at a 640-pixel input size.

### 5.2 Benchmark Results

| Model and configuration | Device | Measured runs | Average latency | Throughput |

|---|---|---:|---:|---:|

| YOLOv8n, 640-pixel input | CPU | 50 | 44.35 ms/image | 22.55 FPS |

| YOLOv8n, 416-pixel input | CPU | 50 | 25.68 ms/image | 38.94 FPS |

| YOLOv8s-World, 640-pixel input | CPU | 50 | 86.27 ms/image | 11.59 FPS |

The results represent the recorded development-machine measurements. Performance may vary with hardware, software versions, system load, and execution conditions.

### 5.3 Results and Interpretation

Reducing the YOLOv8n input size from 640 × 640 to 416 × 416 pixels improved measured inference performance.

The percentage reduction in average latency is:

\[

\text{Latency Reduction} =

\frac{44.35-25.68}{44.35}\times100

\]

\[

\text{Latency Reduction} \approx 42.1\%

\]

The percentage increase in throughput is:

\[

\text{Throughput Increase} =

\frac{38.94-22.55}{22.55}\times100

\]

\[

\text{Throughput Increase} \approx 72.7\%

\]

Using the unrounded benchmark values, the throughput increase is approximately 72.6%.

Therefore, the smaller input size achieved approximately:

- **42.1% lower average latency**

- **72.6% higher throughput**

- **1.73 times the throughput** of the 640-pixel configuration

The 416 × 416 input contains approximately 57.8% fewer pixels than the 640 × 640 input.

Processing fewer pixels can reduce computational requirements and improve inference speed. However, smaller inputs may also reduce detection quality, particularly for small or distant objects.

YOLOv8s-World was slower than YOLOv8n in the recorded CPU benchmark. Nevertheless, its text-prompt-based detection capability makes it useful for experimenting with navigation objects that are not represented by dedicated classes in the baseline model.

These experiments used a repeated image and do not establish live-camera performance or performance on the final robotics hardware. Detection quality at the different input resolutions has not been formally compared.

## 6. Edge Optimization Experiments

### 6.1 Input-Resolution Optimization

The primary optimization experiment compared YOLOv8n inference at 640 × 640 and 416 × 416 pixels.

The smaller input size improved measured throughput on the development machine.

This demonstrates that input resolution is an important parameter when balancing computational cost and inference speed.

However, an effective robotics deployment must also consider detection quality. Further testing should establish whether the smaller input size preserves reliable detection of small or distant obstacles.

### 6.2 ONNX Export

The baseline YOLOv8n model was successfully exported to ONNX format, producing:

`yolov8n.onnx`

However, the ONNX Runtime benchmark terminated in the Windows development environment with exit status `-1073741819` before producing valid performance measurements.

Consequently, no ONNX latency improvement or speedup is claimed.

Further debugging is required before a reliable performance comparison can be made between the original model and its ONNX representation.

### 6.3 Optimization Scope

The input-resolution experiment demonstrated a measurable throughput improvement. ONNX export completed, but ONNX Runtime benchmarking was unsuccessful.

An FP32-versus-INT8 OpenVINO comparison was subsequently performed for inference speed and test-set detection metrics. The results did not show a consistent speed improvement from INT8, and the measured accuracy was low for both models. These results are preliminary; model-to-dataset class compatibility should be verified before interpreting the accuracy comparison as a reliable measure of detection quality.

Pruning was not evaluated in this project. A complete optimization study should compare inference speed, detection quality, model size, and resource usage under consistent experimental conditions.

## 7. OpenVINO Benchmarking

### 7.1 CPU and Intel Iris Xe GPU Comparison

YOLOv8n was exported to OpenVINO format with a 416 × 416 input size.

The CPU and Intel Iris Xe GPU configurations were benchmarked on the same image using five warm-up runs and thirty measured inference runs per device.

### 7.2 Results

| Metric | OpenVINO CPU | OpenVINO Intel Iris Xe GPU |

|---|---:|---:|

| Total time for 30 measured runs | 1.414 s | 1.270 s |

| Average latency | 47.12 ms/image | 42.33 ms/image |

| Throughput | 21.22 FPS | 23.63 FPS |

The Intel Iris Xe GPU delivered approximately 11.4% higher throughput and 10.2% lower average latency than the CPU in this experiment.

These measurements suggest that the GPU configuration performed better under the tested conditions.

However, the results are preliminary and hardware-specific. They do not establish live-camera performance, detection accuracy, or performance on the intended robotics hardware.

### 7.3 FP32 and INT8 Comparison

An OpenVINO experiment compared FP32 and INT8 configurations at a 416 × 416 input size. Speed measurements were recorded separately from the test-set accuracy evaluation.

Inference speed
Model precision	Device	Average latency	Throughput
FP32	CPU	50.85 ms/image	19.67 FPS
INT8	CPU	50.27 ms/image	19.89 FPS
FP32	Intel Iris Xe GPU	39.56 ms/image	25.28 FPS
INT8	Intel Iris Xe GPU	45.67 ms/image	21.90 FPS


On the CPU, INT8 throughput increased by approximately 1.1%. On the Intel Iris Xe GPU, INT8 throughput decreased by approximately 13.4%. The measurements therefore do not demonstrate a consistent speed improvement from quantization across both devices.

The INT8 model directory occupied approximately 3.5 MB, while the original PyTorch checkpoint occupied approximately 6.2 MB. These are different formats, so this is not a controlled, equivalent-format model-size comparison.

Test-set accuracy

Both OpenVINO models were evaluated on the traffic-cone test split configured in data/traffic_cone_public/data_local.yaml, using an image size of 416 × 416 pixels and the CPU device.
Metric	FP32	INT8
Precision	0.05139 (5.14%)	0.04784 (4.78%)
Recall	0.02893 (2.89%)	0.02700 (2.70%)
mAP@50	0.01278 (1.28%)	0.01187 (1.19%)
mAP@50–95	0.00424 (0.42%)	0.00378 (0.38%)


INT8 scored slightly lower than FP32 on all four reported metrics. However, both models produced very low scores on this dataset. Model-to-dataset class compatibility and class mappings must be verified before treating these numbers as a reliable assessment of the detector's actual quality or attributing the difference solely to quantization.

The INT8 calibration set contained only four images. These results should therefore be considered preliminary. A stronger evaluation would use a representative calibration set, confirm that model class labels match the dataset annotations, and run both models under the same evaluation and benchmarking conditions. Pruning was not evaluated.

## 8. Custom Dataset and Fine-Tuning Experiment

### 8.1 Dataset Configuration

A small custom dataset was used for a preliminary cone-detection fine-tuning experiment.

The local dataset directory contains:

- 200 image files in JPG format.

- 200 text annotation files.

- A YAML configuration file defining five classes.

The configured classes are:

1. `blue_cone`

2. `large_orange_cone`

3. `orange_cone`

4. `unknown_cone`

5. `yellow_cone`

The configuration uses the `images` directory for both training and validation.

The dataset's original source has not been independently verified, so it is described here as a custom dataset rather than as BDD100K.

### 8.2 Fine-Tuning Configuration

The preliminary experiment used:

- A custom YOLO-format dataset.

- Five cone-related classes.

- An input image size of 320 pixels.

- One training epoch.

- CPU-based training.

The experiment produced a model checkpoint at:

`runs/detect/runs/navigation/train-3/weights/best.pt`

### 8.3 Observations and Limitations

The resulting model produced false-positive detections and oversized bounding boxes during testing.

The experiment was limited in scope and should not be considered evidence of a reliable, fully trained navigation detector.

Furthermore, the same image directory was configured for training and validation. Consequently, the experiment did not provide an independent assessment of generalization to unseen images.

Further work should include:

- Verifying the dataset's source and annotation quality.

- Separating training, validation, and test images.

- Training for additional epochs with appropriate monitoring.

- Evaluating the model on previously unseen images.

- Measuring precision, recall, and mAP.

- Investigating false positives and bounding-box localization errors.

The baseline and open-vocabulary models remain important components of the project because the preliminary fine-tuning experiment did not establish that the custom-trained model reliably detects the required navigation objects.

## 9. Reproducibility

### 9.1 Environment Setup

Activate the project's Python virtual environment in Windows PowerShell:

```powershell

.\.venv\Scripts\Activate.ps1

```

Install the required dependencies:

```powershell

python -m pip install -r requirements.txt

```

Model weights may need to be downloaded on the first run.

### 9.2 Run Object Detection

Run open-vocabulary detection on the demonstration image:

```powershell

python detect.py --source assets/demo.png --weights yolov8s-worldv2.pt --open-vocabulary --output output/final_demo.jpg

```

The command assumes that `assets/demo.png` exists in the project directory.

### 9.3 Benchmark YOLOv8n at 640 Pixels

```powershell

python benchmark.py --source assets/demo.png --weights yolov8n.pt --device cpu --warmup 5 --runs 50 --imgsz 640

```

### 9.4 Benchmark YOLOv8n at 416 Pixels

```powershell

python benchmark.py --source assets/demo.png --weights yolov8n.pt --device cpu --warmup 5 --runs 50 --imgsz 416

```

### 9.5 Export to ONNX

```powershell

python export_onnx.py --weights yolov8n.pt

```

The export command completed successfully in the recorded experiment. The subsequent ONNX Runtime benchmark did not produce valid performance measurements.

Benchmark results may vary with hardware, software versions, system load, model versions, and execution conditions.

## 10. Limitations

The current prototype has several limitations.
1. Detection reliability: The baseline model does not have dedicated COCO classes for traffic cones and road barriers. Open-vocabulary detection improves flexibility but does not guarantee reliable detection of every target object.
2. Distance accuracy: The monocular distance calculation depends on assumed object dimensions, camera parameters, and bounding-box quality. Real-world accuracy has not been established through measurements at known distances.
3. Dataset size: The preliminary custom fine-tuning experiment used 200 images and five cone-related classes.
4. Independent evaluation: The same image directory was configured for training and validation in the preliminary experiment. Independent validation on unseen data remains necessary.
5. Fine-tuning quality: The resulting custom model produced false positives and oversized bounding boxes.
6. Optimization evaluation: Smaller input resolution improved measured throughput, but its effect on detection quality has not been formally evaluated.
7. Quantization: The FP32-versus-INT8 comparison showed no consistent speed improvement across the tested devices. Test-set metrics were slightly lower for INT8, but both models had very low scores. Class compatibility must be verified before interpreting the accuracy results. The calibration set contained only four images, and pruning was not evaluated.
8. ONNX Runtime: ONNX export succeeded, but the ONNX Runtime benchmark terminated before producing valid measurements.
9. Hardware specificity: The CPU and OpenVINO benchmark results were obtained on the development machine. They do not establish performance on the final robotics hardware.
10. Real-time deployment: Live-camera performance, obstacle-tracking reliability, and deployment on a physical robot have not been validated.
## 11. Future Work

The following improvements would strengthen the system and make it more suitable for practical robotics navigation.

1. Collect and annotate a representative dataset containing traffic cones, road barriers, stop signs, and other relevant road objects.

2. Verify the dataset source and use separate training, validation, and test datasets.

3. Fine-tune the detector for additional epochs and evaluate it on unseen images.

4. Measure precision, recall, and mean Average Precision to quantify detection performance.

5. Compare detection quality at multiple input resolutions to evaluate the trade-off between accuracy and throughput.

6. Calibrate the camera and validate estimated distances against objects placed at known physical distances.

7. Investigate and correct false positives and inaccurate bounding boxes.

8. Debug ONNX Runtime execution and perform a controlled comparison against the original model.

9. Investigate quantization and pruning using representative calibration and evaluation datasets.

10. Compare optimized models using consistent measurements of latency, throughput, model size, resource usage, and detection quality.

11. Investigate object tracking across video frames to improve temporal consistency.

12. Test the complete pipeline using a live camera and the intended robotics hardware.

## 12. Conclusion

This project demonstrates a modular object-detection and monocular distance-estimation pipeline for robotics navigation.

The implementation combines a lightweight YOLOv8n baseline, YOLOv8s-World open-vocabulary detection, bounding-box-based distance estimation, CPU benchmarking, OpenVINO experiments, ONNX export, and a preliminary custom-dataset fine-tuning experiment.

CPU benchmarking showed that reducing YOLOv8n input resolution from 640 × 640 to 416 × 416 pixels improved measured throughput from 22.55 FPS to 38.94 FPS, corresponding to an increase of approximately 72.6% and a reduction in average latency of approximately 42.1%.

The OpenVINO experiment recorded preliminary performance differences between CPU and Intel Iris Xe GPU inference. The FP32-versus-INT8 comparison showed no consistent speed improvement across devices. Test-set accuracy metrics were slightly lower for INT8, but both configurations scored very low; class compatibility should be verified before drawing firm conclusions. Pruning was not evaluated.

The preliminary fine-tuning experiment highlighted the need for better training data, independent validation, and additional model development. Similarly, the distance-estimation calculation requires physical validation before its results can be considered reliable in real-world conditions.

Overall, the project establishes a working prototype and a foundation for further investigation. Additional dataset-based evaluation, camera calibration, optimization testing, and target-hardware validation are necessary before practical robotics deployment.
## Updated Custom-Model Evaluation: FP32 vs INT8

### Experimental Setup
- Dataset: Traffic Cone Detection Dataset
- Classes: blue_cone, large_orange_cone, orange_cone, unknown_cone, yellow_cone
- Test split: 1,000 images and 19,448 annotated objects
- Input resolution: 416 x 416
- Hardware: Intel 11th Gen Core i5-1145G7 CPU
- Runtime: OpenVINO
- Evaluation batch size: 1

### Quantitative Results

| Metric | FP32 | INT8 |
|---|---:|---:|
| Precision | 53.69% | 53.00% |
| Recall | 23.75% | 23.60% |
| mAP@0.50 | 19.66% | 19.20% |
| mAP@0.50:0.95 | 8.82% | 8.61% |
| Inference latency | 9.2 ms/image | 6.4 ms/image |

The INT8 evaluation reported approximately 30.4% lower inference latency than FP32 on this CPU. The reported INT8 OpenVINO model directory size was 3.5 MB. The FP32 checkpoint was approximately 6.2 MB, but these are different formats, so this is not a direct exported-model size comparison.

### Analysis
The INT8 model retained performance close to the FP32 model, with small reductions in precision, recall, and mAP. Its reported inference latency was lower in this evaluation. These results indicate a potential speed-versus-accuracy trade-off on the tested hardware; latency may vary with system load and runtime conditions.

The unknown_cone class remains a major weakness. Its INT8 recall was zero in the reported test evaluation, and its mAP@0.50 was approximately 0.00135. Both models also showed poor overall detection metrics.

### Limitations
The custom model was trained in a limited, quick training experiment. Its current accuracy is not sufficient to claim reliable real-world robotics navigation. INT8 calibration used 200 images, while Ultralytics recommends more than 300 calibration images. Further training, representative calibration, class-level error analysis, and independent validation are needed.

### Conclusion
Both FP32 and INT8 OpenVINO exports were successfully evaluated on the same 1,000-image test split at 416 x 416 resolution. INT8 achieved a reported inference latency of 6.4 ms/image compared with 9.2 ms/image for FP32, while mAP@0.50 decreased from 19.66% to 19.20%. The pipeline is functional, but further model training and validation are required before practical deployment.

