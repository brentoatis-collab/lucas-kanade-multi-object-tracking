> **Applied AI portfolio project by Brent Oatis, MBA**
> Using classical optical flow to analyze motion, trajectories, and activity in operational video.

# Multi-Object Detection and Tracking Using Optical Flow

**Brent Oatis**  
**CST 544 – Artificial Intelligence in Computer Vision**  
**Concordia University St. Paul**  
**Week 5 Assignment**  
**June 14, 2026**

---

## Project Overview

This project implements a multi-object detection and tracking system using the Lucas-Kanade Optical Flow method on the UCF-101 Action Recognition Dataset.

The application detects feature points using the Shi-Tomasi Corner Detection algorithm and tracks object motion across video frames. Motion trajectories are visualized to analyze movement patterns and object behavior within a sports video sequence.

---

## Assignment Objective

The objective of this project is to build an application that:

- Detects multiple moving objects in a video sequence.
- Tracks object movement using Lucas-Kanade Optical Flow.
- Identifies moving objects across frames.
- Visualizes object trajectories.
- Analyzes motion patterns in sports and action videos.

---

## Dataset

### UCF-101 Action Recognition Dataset

The UCF-101 dataset contains over 13,000 realistic action videos spanning 101 action categories.

For this project, a basketball action video was selected to demonstrate player movement tracking and trajectory analysis.

**Dataset Link:**  
https://www.crcv.ucf.edu/data/UCF101.php

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib

---

## Computer Vision Techniques Implemented

### 1. Shi-Tomasi Corner Detection

Shi-Tomasi Corner Detection identifies strong feature points within an image that are suitable for tracking across consecutive frames.

Parameters used:

```python
maxCorners = 100
qualityLevel = 0.3
minDistance = 7
blockSize = 7
```

### 2. Lucas-Kanade Optical Flow

Lucas-Kanade Optical Flow estimates the motion of feature points between video frames by assuming:

- Small movement between frames
- Brightness consistency
- Local neighborhood motion coherence

This enables efficient tracking of moving objects in real time.

### 3. Multi-Object Tracking

Detected feature points are grouped into moving regions and assigned object identifiers.

The system tracks:

- Object position
- Motion direction
- Motion trajectories

across multiple video frames.

### 4. Trajectory Visualization

Motion paths are recorded and plotted to visualize object behavior over time.

Trajectory analysis helps reveal:

- Motion patterns
- Object movement trends
- Activity dynamics

within the video sequence.

---

## Workflow

```text
Load UCF-101 Video
        ↓
Convert Frames to Grayscale
        ↓
Detect Shi-Tomasi Features
        ↓
Compute Lucas-Kanade Optical Flow
        ↓
Track Motion Across Frames
        ↓
Identify Moving Objects
        ↓
Generate Trajectory Visualization
```

---

## Project Outputs

### Original Dataset Frame

- Raw frame extracted from the UCF-101 basketball video.

### Shi-Tomasi Feature Detection

- Displays detected keypoints used for optical flow tracking.

### Multi-Object Tracking

- Displays tracked objects.
- Includes object identifiers.
- Shows optical flow trajectories.

### Trajectory Visualization

- Plots object motion paths over time.
- Uses different colors to distinguish tracked objects.

### Output Video

- Annotated video showing real-time tracking results.

---

## Repository Structure

```text
week5_optical_flow_tracking/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
│
├── videos/
│   └── ucf101_sample.avi
│
├── outputs/
│   ├── original_ucf101_frame.png
│   ├── slide4_shitomasi_keypoints.png
│   ├── optical_flow_tracking.png
│   ├── trajectory_visualization.png
│   └── tracked_output.mp4
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/brentoatis-collab/lucas-kanade-multi-object-tracking.git
```

Navigate to the project folder:

```bash
cd week5-optical-flow-tracking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Execute:

```bash
python main.py
```

The application will:

1. Load the UCF-101 video.
2. Detect Shi-Tomasi feature points.
3. Compute Lucas-Kanade Optical Flow.
4. Track moving objects.
5. Generate tracking visualizations.
6. Save output images and video.

---

## Results and Insights

The system successfully demonstrated:

- Multi-object tracking using optical flow.
- Reliable feature point tracking across frames.
- Object identification using motion grouping.
- Trajectory visualization for motion analysis.
- Real-time sports movement tracking capabilities.

This project illustrates how optical flow techniques can be applied in sports analytics, surveillance, autonomous systems, and human activity recognition.

---

## References

Bouguet, J.-Y. (2001). *Pyramidal implementation of the Lucas-Kanade feature tracker*. Intel Corporation.

OpenCV Team. (2023). *OpenCV: Open Source Computer Vision Library*. https://opencv.org

Shi, J., & Tomasi, C. (1994). *Good features to track*. Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 593–600. https://doi.org/10.1109/CVPR.1994.323794

Soomro, K., Zamir, A. R., & Shah, M. (2012). *UCF101: A dataset of 101 human actions classes from videos in the wild*. University of Central Florida. https://www.crcv.ucf.edu/data/UCF101.php

Berrios, I. (2024). *Motion Detection: Part 2 – Optical Flow*. Medium. https://medium.com/@itberrios6/introduction-to-motion-detection-part-2-6ec3d6b385d4

Biswas, D. (2022). *Lucas-Kanade Method for Optical Flow*. Medium. https://dibyendu-biswas.medium.com/lucas-kanade-method-for-optical-flow-87ea48dd3e69

---

## Author

**Brent Oatis**  
MBA | Artificial Intelligence Concentration  
Concordia University St. Paul  
CST 544 – Artificial Intelligence in Computer Vision
