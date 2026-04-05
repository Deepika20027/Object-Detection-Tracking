
# Object-Detection-Tracking

This project implements a real-time object detection and tracking system using YOLOv8 and DeepSORT. It can detect and track multiple objects such as cars and people in video streams.

# Click below to run the project directly:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deepika20027/Object-Detection-Tracking/blob/main/RPC%20(2).ipynb)

# Features
- Object detection using YOLOv8
- Multi-object tracking with DeepSORT
- Unique ID assignment for each object
- Video processing and output generation
- Model evaluation dashboard
- 
# Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- DeepSORT
- Google Colab
  
# How to Run
- Open the notebook in Google Colab
- Install dependencies using requirements.txt
- Upload a video
- Click "Process Video"
  
# Output
- Bounding boxes on objects
- Object tracking with IDs
- Processed output video

## screenshot

[![Watch Demo](thumbnail.png)](https://github.com/Deepika20027/Object-Detection-Tracking/raw/main/download.mp4)

## Project Structure

```
object-detection-tracking/
│
├── RPC.ipynb              # Main Google Colab notebook (detection + tracking)
├── requirements.txt       # List of required Python libraries
├── README.md              # Project documentation
│
├── outputs/               # (Generated after running project)
│   ├── output_tracked.mp4 # Processed video with detection
│   ├── tracking_data.csv  # Detection results
│   └── report.html        # Generated report
```

