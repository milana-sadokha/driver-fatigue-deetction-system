# 🚗💤 Project of an driver fatigue detection system in electric vehicles using image and audio analysis

> A real-time multimodal system for detecting driver fatigue, combining facial image analysis with acoustic (audio) analysis. Designed with electric vehicles in mind — optimized for low energy consumption and minimal load on the 12V electrical system.

This repository contains the technical implementation and code accompanying the thesis — it does not include the full thesis text.

--- 

## Why this project 

Fatigue is a factor in 10-20% of road accidents. Most commercial driver monitoring systems rely on a single data source (usually just a camera) and fail under poor lighting or when the face is partially occluded. This system combines two independent modalities (vision + audio) into one fault-tolerant pipeline, while keeping the model lightweight enough to run in real time on hardware with limited compute.

## How it works 

### Vision pipeline

* **MediaPipe Face Mesh** locates 468 facial landmarks in real time.

  > **Design choice:** During the initial phase, I evaluated standard Haar Cascade classifiers against MediaPipe. Empirical testing demonstrated that MediaPipe offers significantly higher tracking stability, especially during head rotations and under the variable, low-light conditions typical of a vehicle cabin.

* Eye and mouth regions (ROI) are cropped based on landmark indices and resized to 64×64 px grayscale.
* **Two custom CNNs** independently classify: eyelid state (open/closed) and yawning symptoms.
* **Custom Heuristic for Head Pose:** Instead of relying on computationally expensive 3D pose estimation algorithms, I developed a lightweight 2D geometric heuristic. Head pitch and yaw are calculated using simple mathematical proportions between specific landmarks. This significantly reduces CPU load while effectively detecting micro-sleeps (head dropping) and loss of focus.

### Audio pipeline 

* Microphone signal is split into 3-second segments, filtered by an RMS energy threshold (drops silence and clipped/overloaded audio).
* Converted to a Mel-spectrogram (128 filters) → classified by a CNN (yawn / no yawn).

### Fusion and decision logic 

* **Late (decision-level) fusion** — each pipeline runs independently, and results are combined only at the decision stage.
* **Priority hierarchy:** `ALERT` (eyes closed / critical head drop) → `DROWSY` (sustained head tilt) → `WARNING` (yawning / lack of focus).
* Time-window buffers filter out false positives from brief, natural head movements.
* **Multi-threaded architecture** — audio analysis runs on a background thread and never blocks video processing.

---

## Why a custom CNN instead of a pretrained model (VGG16, etc.)

I ran a direct comparison between the custom architecture and VGG16 on the same task (eyelid state classification):

| Metric | Custom model | VGG16 |
| :--- | :--- | :--- |
| **Size** | 7.87 MB | 62.22 MB |
| **Parameters** | 683,716 | 15,240,260 |
| **Inference time** | 2.96 ms | 6.16 ms |
| **Accuracy** | 99.40% | 97.37% |

The smaller, purpose-built model turned out to be both faster and more accurate for this specific task — VGG16 is designed for classifying thousands of classes on 224×224 images, which is unnecessary complexity for binary classification of 64×64 crops. In an EV context, a smaller model also means lower energy draw and less heat dissipation.

--- 

## Custom head-pose heuristic 

Instead of computationally expensive 3D head-pose estimation, head position is derived directly from the geometric proportions of MediaPipe landmarks (nose, forehead, chin, face edges):


$$Pitch = \frac{y_{nose} - y_{forehead}}{|y_{chin} - y_{forehead}|}$$
*(downward head drop)*

$$Yaw = \frac{x_{nose} - x_{right\_edge}}{|x_{left\_edge} - x_{right\_edge}|}$$
*(sideways gaze)*

Dividing by the full face height/width keeps the metrics stable regardless of the driver's distance from the camera — without this normalization, the same raw values would mean different things depending on how close the driver sits to the lens.


---

## Why VGG16 as the benchmark

VGG16 was deliberately chosen as a widely recognized, standard image classification benchmark — not because it was a natural candidate for this task, but precisely to highlight the contrast: a large, general-purpose architecture (designed for thousands of classes, 224×224 RGB images) versus a small, purpose-built model for a narrow, specific task (binary classification, 64×64, grayscale). The comparison (table below) confirms that for this type of task, model size does not translate into better quality — if anything, the opposite.


| Metric | Custom model | VGG16 |
| :--- | :--- | :--- |
| **Size** | 7.87 MB | 62.22 MB |
| **Parameters** | 683,716 | 15,240,260 | 
| **Inference time** | 2.96 ms | 6.16 ms |
| **Accuracy** | 99.40% | 97.37% | 

---

## Dataset sizes 

| Classifier | Train | Validation | Test | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Eyelid state + yawn(visual)** | 6,382 | 1,823 | 914 | 9,119 |
| **Yawn (audio)** | - | - | - | - |

The 70/20/10 train/validation/test split for the vision pipeline ensures the final model evaluation happens on data that took no part in weight optimization at any stage.

---

## Results 

| Classifier | Precision | Recall | F1-score | Test set |
| :--- | :--- | :--- | :--- | :--- |
| **Eyelid state** | 1.00 | 1.00 | 1.00 | 400 images |
| **Yawn (visual)** | 0.98 | 0.98 | 0.98 | 514 images |
| **Yawn (audio)** | 0.98 | 0.98 | 0.98 | 397 recordings |

**Honest note:** the 100% result for the eyelid classifier most likely reflects limited variance in the public dataset used (Eyes Open/Closed, Kaggle) rather than real-world performance in a vehicle cabin. Under varied lighting conditions and with different face types/eyewear, I'd expect a lower — though still strong — accuracy. This is one of the planned next steps for further testing.

---

## Treining curves

---

## Interface preview 

The overlay shows live classification scores, head-pose values, and buffer counters for each detected state: 


---

## Tech stack 

`Python` · `TensorFlow / Keras` · `OpenCV` · `MediaPipe` · `librosa` · `sounddevice` · `NumPy`

## Repository structure 

```text
.
├── app/            # Real-time demo application (camera + mic → live alerts)
├── vision/         # Eye state + yawn CNN training
├── audio/          # Audio feature extraction + yawn CNN training
├── experiments/    # Haar Cascades vs MediaPipe comparison
└── assets/         # Diagrams, screenshots, training curves
```

## Setup


```text
pip install -r requirements.txt
python app/main.py
```

---

## Limitations and future work


assets/         diagrams, screenshots, training curves
