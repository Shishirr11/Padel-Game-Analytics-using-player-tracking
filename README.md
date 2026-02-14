## Introduction

This project is a watch-and-analyse pipeline for padel: you give it a short match clip, and it tries to follow the players and the ball frame by frame, then prints a simple summary at the end (how far everyone moved and their average speed).
The workflow is hands-on: on the very first frame, you click the four **Court Corners** so the code understands where the court is in the video. After that, it runs automatically and writes an annotated output video.

---

## Data and Preprocessing

### Data
- A single sample video is included here:
  - `data/rally.mp4`

Outputs are written to:
- `output/tracked_rally.mp4` (annotated video)

### Preprocessing 
**Court setup:**
- The program opens the first frame and asks you to click 4 points in this order:
  1) top-left  
  2) top-right  
  3) bottom-right  
  4) bottom-left  

**Players:**
- Each frame is passed to YOLO, and only class `0` (person) detections are kept.
- The code assigns stable IDs across frames using IoU matching (so “Player 1” doesn’t keep changing).

**Ball(short window):**
- The ball tracker looks at a rolling buffer of **3 frames** at a time.
- Those 3 frames are resized to **640×360**, normalized to `[0,1]`, and stacked together so the model sees motion (3 frames → 9 channels).
- The model outputs a heatmap; if the heatmap is weak (max < 0.5), the frame is treated as “ball not found”.
- If it’s found, the peak heatmap location is mapped back to the original frame size.

**Projection to court-coordinates:**
- Once players and ball are found in pixel space, their positions are projected onto a simple court coordinate system using a homography (so distance/speed can be calculated in consistent units).

**Smoothing (jitter):**
- Both player coordinates and ball coordinates are averaged over the last **5** positions to calm down frame-to-frame noise.

---

## Model Architecture

This pipeline is basically **two trackers running in parallel**, then one analytics step at the end:

### 1) Player detection: YOLOv8 (Ultralytics)
- Model: `yolov8n.pt`
- Role: find player bounding boxes each frame.
- Output used downstream:
  - bounding boxes → converted to player “center points”
  - IDs assigned with IoU matching so the same person keeps the same ID

### 2) Ball tracking: TrackNet-style heatmap model (BallTrackerNet)
- File: `tracknet_architecture.py`
- Input: **3 consecutive RGB frames** stacked into a 9-channel tensor
- Backbone style: encoder → bottleneck → decoder (U-Net-ish)
  - conv blocks + pooling down -> upsmapling

### 3) Court projection: Homography mapping
- File: `projection.py`
- Takes your clicked corners and maps pixel coordinates into a flat “court plane”.

### 4) Analytics summary
- File: `analytics.py`
- For each frame, it stores:
  - `players: [{id, coords}]`
  - `ball: (x, y)` or None
- Metrices :
  - distance  
  - average speed
  - ball distance 
  - ball average

---

## Prerequisites

### Python version / Libraries 
- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLO)

Install:
```bash
pip install opencv-python numpy torch ultralytics
```

### Required Models
a TrackNet weight file here:
- `tracknet_model_best.pt`

It is not included (the same folder as Models) before running.

YOLO weights (`yolov8n.pt`) are typically downloaded automatically by Ultralytics on first run.

---

## How to run it 

1) Clone or download and unzip the project and open a terminal inside the project folder.

2) Make sure the video exists:
- `data/rally.mp4`

3) Add your TrackNet weights:
- place `tracknet_model_best.pt` in the root folder (next to `main.py`)

4) Run:
```bash
  python main.py
```
5.	Click the court corners when the first frame pops up:

    - Top-left → Top-right → Bottom-right → Bottom-left

6.	Check outputs:

	-	Annotated video: `output/tracked_rally.mp4`
	- Console summary: distances + average speeds for players and ball
