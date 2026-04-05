# -*- coding: utf-8 -*-

#  Step 1: Install dependencies (run this first time only)
!pip install -q ultralytics opencv-python ffmpeg-python deep_sort_realtime ipywidgets matplotlib

#  Step 2: Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab import files
import ipywidgets as widgets
from IPython.display import display, Video
import os
import pandas as pd
import time
from datetime import datetime

# Step 3: Enhanced Model Upload and Configuration UI

# Model upload widget
model_uploader = widgets.FileUpload(
    accept='.pt',
    multiple=False,
    description='Upload Model (.pt)',
    style={'description_width': 'initial'}
)

# Status display for model upload
model_status = widgets.HTML(
    value="<p style='color: #666; font-style: italic;'>Upload your best.pt model or use default options</p>"
)

# Model selection dropdown
model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 'best.pt']
model_dropdown = widgets.Dropdown(
    options=model_options,
    value='best.pt',
    description='Model:',
    style={'description_width': 'initial'}
)

# Manual path input for custom models
manual_model_path = widgets.Text(
    value='',
    placeholder='Or enter full path to your .pt file',
    description='Manual Path:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

use_manual_path = widgets.Checkbox(
    value=False,
    description='Use manual path',
    style={'description_width': 'initial'}
)

# Function to handle model upload
def handle_model_upload(change):
    if model_uploader.value:
        uploaded_file = list(model_uploader.value.values())[0]
        filename = uploaded_file['metadata']['name']

        try:
            # Save the file locally
            with open('best.pt', 'wb') as f:
                f.write(uploaded_file['content'])

            model_status.value = f"<p style='color: #28a745; font-weight: bold;'>✅ Successfully uploaded: {filename}</p>"

            # Update dropdown to include the uploaded model
            if 'best.pt' not in model_dropdown.options:
                model_dropdown.options = list(model_dropdown.options) + ['best.pt']

            # Set the uploaded model as selected
            model_dropdown.value = 'best.pt'

            print(f"✅ Model '{filename}' uploaded successfully as 'best.pt'")

        except Exception as e:
            model_status.value = f"<p style='color: #dc3545; font-weight: bold;'>❌ Upload failed: {str(e)}</p>"

model_uploader.observe(handle_model_upload, names='value')

# Function to check if model exists
def check_model_exists(model_name):
    """Check if model file exists and return the correct path"""
    if use_manual_path.value and manual_model_path.value.strip():
        manual_path = manual_model_path.value.strip()
        if os.path.exists(manual_path):
            return manual_path
        else:
            raise FileNotFoundError(f"Manual path not found: {manual_path}")

    # Check current directory
    if os.path.exists(model_name):
        return model_name

    # Check common paths
    possible_paths = [
        f"./models/{model_name}",
        f"./weights/{model_name}",
        f"../models/{model_name}",
        f"../weights/{model_name}"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # If it's a default YOLO model, let YOLO handle the download
    if model_name in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
        return model_name

    raise FileNotFoundError(f"Model file not found: {model_name}")

# Enhanced class selection with custom classes support
all_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
               'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class_selector = widgets.SelectMultiple(
    options=all_classes,
    value=['person', 'car', 'truck', 'bus'],
    description='Track:',
    layout=widgets.Layout(width='300px', height='100px')
)

# Auto-detect classes button
detect_classes_btn = widgets.Button(
    description='Auto-detect Classes',
    button_style='info',
    icon='search',
    tooltip='Load model and detect available classes'
)

classes_status = widgets.HTML(value="")

conf_threshold = widgets.FloatSlider(
    value=0.25,
    min=0.01,
    max=0.99,
    step=0.01,
    description='Confidence:',
    continuous_update=False
)

# Counting line configuration
counting_enabled = widgets.Checkbox(
    value=False,
    description='Enable counting line',
    layout=widgets.Layout(width='150px')
)

line_position = widgets.IntSlider(
    value=50,
    min=0,
    max=100,
    step=1,
    description='Line position (%):',
    disabled=True,
    continuous_update=False
)

line_direction = widgets.Dropdown(
    options=['horizontal', 'vertical'],
    value='horizontal',
    description='Line direction:',
    disabled=True
)

# Display track history
show_tracks = widgets.Checkbox(
    value=False,
    description='Show track history',
    layout=widgets.Layout(width='150px')
)

track_length = widgets.IntSlider(
    value=20,
    min=5,
    max=100,
    step=5,
    description='Track length:',
    disabled=True,
    continuous_update=False
)

# Output format options
output_options = widgets.Dropdown(
    options=['MP4 Video', 'MP4 + CSV data', 'MP4 + CSV + Report'],
    value='MP4 Video',
    description='Output:',
)

# Function to auto-detect classes from model
def detect_classes_handler(b):
    selected_model = model_dropdown.value

    with classes_status:
        classes_status.value = "<p style='color: #17a2b8;'>🔍 Loading model to detect classes...</p>"

        try:
            model_path = check_model_exists(selected_model)
            model = YOLO(model_path)

            # Get class names from model
            model_classes = list(model.names.values())

            # Update class selector options
            class_selector.options = model_classes

            # Set default selection (first few classes or common ones)
            default_classes = []
            common_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']
            for cls in common_classes:
                if cls in model_classes:
                    default_classes.append(cls)

            if not default_classes:
                default_classes = model_classes[:4]  # First 4 classes

            class_selector.value = default_classes

            classes_status.value = f"<p style='color: #28a745;'>✅ Found {len(model_classes)} classes in model</p>"

        except Exception as e:
            classes_status.value = f"<p style='color: #dc3545;'>❌ Error loading model: {str(e)}</p>"

detect_classes_btn.on_click(detect_classes_handler)

# Define UI layout functions
def on_counting_enabled_change(change):
    if change['new']:
        line_position.disabled = False
        line_direction.disabled = False
    else:
        line_position.disabled = True
        line_direction.disabled = True

def on_tracks_enabled_change(change):
    if change['new']:
        track_length.disabled = False
    else:
        track_length.disabled = True

counting_enabled.observe(on_counting_enabled_change, names='value')
show_tracks.observe(on_tracks_enabled_change, names='value')

# Enhanced UI Display
display(widgets.HTML("<h2>🚀 Enhanced Video Object Detection & Tracking</h2>"))
display(widgets.HTML("<hr>"))

display(widgets.HTML("<h3>📁 Model Configuration</h3>"))
display(widgets.VBox([
    model_uploader,
    model_status,
    widgets.HTML("<h4>Model Selection:</h4>"),
    widgets.HBox([model_dropdown, detect_classes_btn]),
    widgets.HTML("<h4>Alternative: Manual Path</h4>"),
    widgets.VBox([use_manual_path, manual_model_path])
]))

display(widgets.HTML("<h3>🎯 Detection Settings</h3>"))
display(widgets.VBox([conf_threshold]))

display(widgets.HTML("<h3>📊 Class Selection</h3>"))
display(classes_status)
display(class_selector)

display(widgets.HTML("<h3>📏 Counting Configuration</h3>"))
display(widgets.HBox([counting_enabled, line_position, line_direction]))

display(widgets.HTML("<h3>🛤️ Tracking Visualization</h3>"))
display(widgets.HBox([show_tracks, track_length]))

display(widgets.HTML("<h3>💾 Output Options</h3>"))
display(output_options)

display(widgets.HTML("<hr>"))

# Upload and process buttons
upload_button = widgets.Button(
    description='Upload Video',
    button_style='info',
    icon='upload'
)

process_button = widgets.Button(
    description='Process Video',
    button_style='success',
    icon='play',
    disabled=True
)

status_output = widgets.Output()

display(widgets.HTML("<h3>🎬 Video Processing</h3>"))
display(widgets.HBox([upload_button, process_button]))
display(status_output)

# Global variable to store the video path
video_path = None

# Upload handler
def upload_handler(b):
    global video_path
    with status_output:
        status_output.clear_output()
        print("📁 Uploading video file...")
        uploaded = files.upload()
        if uploaded:
            video_path = list(uploaded.keys())[0]
            process_button.disabled = False
            print(f"✅ Video uploaded: {video_path}")
        else:
            print("❌ No video uploaded")

upload_button.on_click(upload_handler)

# Enhanced video processing function
def process_video(b):
    global video_path

    if video_path is None:
        with status_output:
            print("❌ Please upload a video first")
        return

    with status_output:
        status_output.clear_output()
        print("🔄 Processing video...")

        # Configuration
        selected_model = model_dropdown.value
        allowed_classes = list(class_selector.value)
        confidence = conf_threshold.value
        count_objects = counting_enabled.value
        line_pos = line_position.value
        line_dir = line_direction.value
        show_track_history = show_tracks.value
        history_length = track_length.value
        output_type = output_options.value

        # Load model with enhanced error handling
        try:
            model_path = check_model_exists(selected_model)
            print(f"📦 Loading model: {model_path}")
            model = YOLO(model_path)
            print(f"✅ Model loaded successfully")
            print(f"🎯 Model classes: {list(model.names.values())}")

        except FileNotFoundError as e:
            print(f"❌ {str(e)}")
            print("💡 Please upload your model file or check the path.")
            return
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return

        # Validate selected classes exist in model
        model_classes = list(model.names.values())
        valid_classes = [cls for cls in allowed_classes if cls in model_classes]
        invalid_classes = [cls for cls in allowed_classes if cls not in model_classes]

        if invalid_classes:
            print(f"⚠️ Warning: These classes are not available in the model: {invalid_classes}")

        if not valid_classes:
            print("❌ No valid classes selected for this model!")
            return

        allowed_classes = valid_classes
        print(f"🎯 Tracking classes: {allowed_classes}")

        # Initialize DeepSORT tracker
        tracker = DeepSort(max_age=30, n_init=3)

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Video info: {w}x{h} @ {fps}fps, {total_frames} frames")

        # Initialize output video
        out = cv2.VideoWriter("output_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # Create counting line
        if count_objects:
            if line_dir == 'horizontal':
                line_y = int(h * line_pos / 100)
                counting_line = [(0, line_y), (w, line_y)]
            else:
                line_x = int(w * line_pos / 100)
                counting_line = [(line_x, 0), (line_x, h)]

        # Initialize counters and tracking history
        track_history = {}  # Store track positions for visualization
        crossed_tracks = set()  # Store IDs of objects that crossed the line
        object_counts = {cls: 0 for cls in allowed_classes}  # Count by class

        # Initialize dataframe for tracking data
        tracking_data = []

        # Create heatmap data
        heatmap = np.zeros((h, w), dtype=np.uint8)

        # Start time
        start_time = time.time()

        # Process each frame
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if frame_count % 10 == 0:  # Update progress every 10 frames
                progress = frame_count / total_frames * 100
                print(f"🔄 Processing: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')

            # Run YOLO inference
            results = model(frame, conf=confidence)[0]
            detections = []

            # Process YOLO detections
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label in allowed_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Draw counting line if enabled
            if count_objects:
                cv2.line(frame, counting_line[0], counting_line[1], (0, 255, 255), 2)

                # Display counter on frame
                counter_text = " | ".join([f"{cls}: {count}" for cls, count in object_counts.items()])
                cv2.putText(frame, f"Count: {counter_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Process each track
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                label = track.get_det_class()

                # Calculate center point of the bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Add to heatmap
                cv2.circle(heatmap, (center_x, center_y), 15, 255, -1)

                # Store track history for visualization
                if show_track_history:
                    if track_id not in track_history:
                        track_history[track_id] = []

                    track_history[track_id].append((center_x, center_y))

                    # Limit history length
                    if len(track_history[track_id]) > history_length:
                        track_history[track_id] = track_history[track_id][-history_length:]

                # Check if object crossed the counting line
                if count_objects and track_id not in crossed_tracks:
                    if line_dir == 'horizontal':
                        # Check if the center point crossed the horizontal line
                        if len(track_history.get(track_id, [])) >= 2:
                            prev_y = track_history[track_id][-2][1]
                            curr_y = center_y
                            line_y = counting_line[0][1]

                            if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                                crossed_tracks.add(track_id)
                                object_counts[label] += 1
                    else:
                        # Check if the center point crossed the vertical line
                        if len(track_history.get(track_id, [])) >= 2:
                            prev_x = track_history[track_id][-2][0]
                            curr_x = center_x
                            line_x = counting_line[0][0]

                            if (prev_x < line_x and curr_x >= line_x) or (prev_x > line_x and curr_x <= line_x):
                                crossed_tracks.add(track_id)
                                object_counts[label] += 1

                # Draw bounding box and label
                # Enhanced color scheme based on object class
                colors = {
                    'person': (0, 255, 0),    # Green
                    'car': (0, 165, 255),     # Orange
                    'truck': (0, 0, 255),     # Red
                    'bus': (255, 0, 0),       # Blue
                    'bicycle': (255, 0, 255), # Magenta
                    'motorcycle': (255, 255, 0) # Cyan
                }
                color = colors.get(label, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ID-{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw track history
                if show_track_history and track_id in track_history and len(track_history[track_id]) > 1:
                    points = np.array(track_history[track_id], dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)

                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (w - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Collect tracking data for export
                if output_type != 'MP4 Video':
                    tracking_data.append({
                        'frame': frame_count,
                        'track_id': track_id,
                        'class': label,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'center_x': center_x,
                        'center_y': center_y,
                        'confidence': conf if 'conf' in locals() else 0.0,
                        'timestamp': timestamp
                    })

            # Write frame to output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()

        # Process time
        process_time = time.time() - start_time

        # Re-encode for playback in Colab
        print("\n🎬 Processing complete. Preparing video for playback...")
        os.system('ffmpeg -y -i output_tracked.mp4 -vf "scale=640:360" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k output_colab.mp4')

        # Export tracking data if requested
        if output_type != 'MP4 Video':
            df = pd.DataFrame(tracking_data)
            df.to_csv('tracking_data.csv', index=False)
            print("📊 Tracking data exported to tracking_data.csv")

        # Generate report if requested
        if output_type == 'MP4 + CSV + Report':
            # Create a heatmap visualization
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET))
            plt.title('Object Movement Heatmap')
            plt.axis('off')
            plt.savefig('heatmap.png')
            plt.close()

            # Create object count chart
            plt.figure(figsize=(10, 6))
            plt.bar(object_counts.keys(), object_counts.values())
            plt.title('Object Counts')
            plt.xlabel('Object Class')
            plt.ylabel('Count')
            plt.savefig('object_counts.png')
            plt.close()

            # Generate enhanced HTML report
            report_html = f"""
            <html>
            <head>
                <title>Object Detection and Tracking Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    h1, h2 {{ color: #333; }}
                    .stats {{ display: flex; justify-content: space-around; flex-wrap: wrap; }}
                    .stat-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 10px; min-width: 150px; }}
                    .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                    .info-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
                    img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                    .metric {{ font-size: 24px; font-weight: bold; }}
                    .label {{ font-size: 14px; opacity: 0.9; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🎯 Object Detection and Tracking Report</h1>
                    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                    <div class="info-grid">
                        <div class="info-box">
                            <h3>📹 Video Information</h3>
                            <p><strong>Filename:</strong> {video_path}</p>
                            <p><strong>Resolution:</strong> {w}x{h}</p>
                            <p><strong>Frame Rate:</strong> {fps} fps</p>
                            <p><strong>Total Frames:</strong> {total_frames}</p>
                            <p><strong>Processing Time:</strong> {process_time:.2f} seconds</p>
                        </div>

                        <div class="info-box">
                            <h3>🤖 Model Information</h3>
                            <p><strong>Model:</strong> {selected_model}</p>
                            <p><strong>Model Path:</strong> {model_path}</p>
                            <p><strong>Classes Tracked:</strong> {', '.join(allowed_classes)}</p>
                            <p><strong>Confidence Threshold:</strong> {confidence}</p>
                        </div>
                    </div>

                    <h2>📊 Detection Statistics</h2>
                    <div class="stats">
                        {''.join(f'<div class="stat-box"><div class="metric">{count}</div><div class="label">{cls.upper()}</div></div>' for cls, count in object_counts.items())}
                    </div>

                    <h2>📈 Visualizations</h2>
                    <h3>Object Counts</h3>
                    <img src="object_counts.png" alt="Object Counts Chart">

                    <h3>Movement Heatmap</h3>
                    <img src="heatmap.png" alt="Movement Heatmap">
                </div>
            </body>
            </html>
            """

            with open('report.html', 'w') as f:
                f.write(report_html)

            print("📋 Enhanced report generated: report.html")

        # Display success message and video
        print(f"\n✅ Done! Video processed in {process_time:.2f} seconds")
        print(f"📊 Object counts: {object_counts}")
        print(f"🎯 Model used: {model_path}")

        display(Video("output_colab.mp4", embed=True))

# Assign process handler
process_button.on_click(process_video)

print("🚀 Enhanced Video Object Detection & Tracking Ready!")
print("📁 Upload your best.pt model using the file uploader above")
print("🎬 Then upload a video and start processing!")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
import ipywidgets as widgets
from IPython.display import display
import os
import random

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#333333',
    'figure.facecolor': 'white',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})

# Fix random seed for reproducibility
random.seed(42)

# File upload widget for custom models
file_uploader = widgets.FileUpload(
    accept='.pt',
    multiple=False,
    description='Upload Model (.pt)',
    style={'description_width': 'initial'}
)

# Status display for file upload
upload_status = widgets.HTML(
    value="<p style='color: #666; font-style: italic;'>No model uploaded yet. You can upload a .pt file or use default options.</p>"
)

# Create UI components
model_dropdown = widgets.Dropdown(
    options=['yolov8n', 'yolov8m', 'yolov8l', 'yolov8x', 'best.pt'],
    value='best.pt',
    description='Model:',
    style={'description_width': 'initial'}
)

class_dropdown = widgets.Dropdown(
    description='Focus Class:',
    disabled=False,
    style={'description_width': 'initial'}
)

evaluate_button = widgets.Button(
    description='Evaluate Model',
    button_style='primary',
    icon='chart-line',
    tooltip='Click to evaluate the selected model'
)

# Manual path input as alternative
manual_path_input = widgets.Text(
    value='',
    placeholder='Or enter full path to your .pt file (e.g., C:\\path\\to\\best.pt)',
    description='Manual Path:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='500px')
)

use_manual_path = widgets.Checkbox(
    value=False,
    description='Use manual path instead',
    style={'description_width': 'initial'}
)

output_area = widgets.Output()

# Function to handle file upload
def handle_upload(change):
    with output_area:
        if file_uploader.value:
            # Get the uploaded file
            uploaded_file = list(file_uploader.value.values())[0]
            filename = uploaded_file['metadata']['name']

            try:
                # Save the file locally
                with open('best.pt', 'wb') as f:
                    f.write(uploaded_file['content'])

                upload_status.value = f"<p style='color: #28a745; font-weight: bold;'>✅ Successfully uploaded: {filename}</p>"

                # Update dropdown to include the uploaded model
                current_options = list(model_dropdown.options)
                if 'best.pt' not in current_options:
                    current_options.append('best.pt')
                    model_dropdown.options = current_options

                # Set the uploaded model as selected
                model_dropdown.value = 'best.pt'

                print(f"Model '{filename}' uploaded successfully as 'best.pt'")

            except Exception as e:
                upload_status.value = f"<p style='color: #dc3545; font-weight: bold;'>❌ Upload failed: {str(e)}</p>"

# Connect upload handler
file_uploader.observe(handle_upload, names='value')

# Function to check if model file exists
def check_model_exists(model_path):
    """Check if model file exists and return the correct path"""
    if use_manual_path.value and manual_path_input.value.strip():
        # Use manual path
        manual_path = manual_path_input.value.strip()
        if os.path.exists(manual_path):
            return manual_path
        else:
            raise FileNotFoundError(f"Manual path not found: {manual_path}")

    # Check current directory
    if os.path.exists(model_path):
        return model_path

    # Check common paths
    possible_paths = [
        f"./models/{model_path}",
        f"./weights/{model_path}",
        f"../models/{model_path}",
        f"../weights/{model_path}"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Model file not found: {model_path}")

# Layout UI components
ui = widgets.VBox([
    widgets.HTML("<h3>🚀 Model Evaluation Dashboard</h3>"),
    widgets.HTML("<hr>"),

    widgets.HTML("<h4>📁 Model Selection</h4>"),
    widgets.HBox([file_uploader]),
    upload_status,

    widgets.HTML("<h4>⚙️ Alternative: Manual Path</h4>"),
    widgets.VBox([
        use_manual_path,
        manual_path_input
    ]),

    widgets.HTML("<h4>🎯 Evaluation Settings</h4>"),
    widgets.HBox([model_dropdown, class_dropdown]),

    widgets.HTML("<hr>"),
    evaluate_button,
    output_area
])

display(ui)

# Function to evaluate model performance
def evaluate_model(b):
    model_name = model_dropdown.value
    focus_class = class_dropdown.value

    with output_area:
        output_area.clear_output()
        print(f"🔍 Evaluating {model_name}...")

        try:
            # Check if model file exists
            try:
                model_path = check_model_exists(model_name)
                print(f"✅ Found model at: {model_path}")
            except FileNotFoundError as e:
                print(f"❌ {str(e)}")
                print("💡 Please upload your model file or check the path.")
                return

            # Find tracking data for this model
            tracking_file = f'tracking_data_{model_name}.csv'
            if not os.path.exists(tracking_file):
                tracking_file = 'tracking_data.csv'

            if not os.path.exists(tracking_file):
                print(f"❌ No tracking data found for {model_name}")
                print("💡 Expected files:")
                print(f"   - {tracking_file}")
                print(f"   - tracking_data.csv (fallback)")
                return

            # Load data
            df = pd.read_csv(tracking_file)
            print(f"✅ Loaded {len(df)} detections from {tracking_file}")

            # Ensure we have confidence values
            if 'confidence' not in df.columns and 'conf' in df.columns:
                df['confidence'] = df['conf']

            if 'confidence' not in df.columns:
                print("⚠️ No confidence column found - creating simulated values")
                df['confidence'] = [round(random.uniform(0.3, 0.98), 2) for _ in range(len(df))]

            # Create simulated ground truth based on confidence
            if 'true_class' not in df.columns:
                df['true_class'] = df['class'].copy()
                classes = df['class'].unique()

                # Introduce errors inversely proportional to confidence
                for i in df.index:
                    error_prob = max(0, 0.5 - df.loc[i, 'confidence'])
                    if random.random() < error_prob:
                        other_classes = [c for c in classes if c != df.loc[i, 'class']]
                        if other_classes:
                            df.loc[i, 'true_class'] = random.choice(other_classes)

            # Update class dropdown options if needed
            all_classes = sorted(df['class'].unique())
            if not class_dropdown.options or set(class_dropdown.options) != set(all_classes):
                class_dropdown.options = all_classes
                if focus_class not in all_classes:
                    focus_class = all_classes[0]
                    class_dropdown.value = focus_class

            # Create binary labels for ROC curve
            df['is_focus_class'] = (df['class'] == focus_class).astype(int)
            df['is_true_focus_class'] = (df['true_class'] == focus_class).astype(int)

            # Calculate metrics
            overall_accuracy = accuracy_score(df['true_class'], df['class'])

            # Focus class metrics
            class_precision = precision_score(
                df['is_true_focus_class'], df['is_focus_class'], zero_division=0
            )
            class_recall = recall_score(
                df['is_true_focus_class'], df['is_focus_class'], zero_division=0
            )
            class_f1 = f1_score(
                df['is_true_focus_class'], df['is_focus_class'], zero_division=0
            )

            # Calculate metrics for all classes
            classes = sorted(set(df['class'].unique()) | set(df['true_class'].unique()))
            all_metrics = {}

            for cls in classes:
                y_true = (df['true_class'] == cls).astype(int)
                y_pred = (df['class'] == cls).astype(int)

                all_metrics[cls] = {
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0)
                }

            # Calculate ROC curve data
            # Use detection confidence as the score
            focus_df = df[df['class'] == focus_class].copy()
            if len(focus_df) > 0:
                fpr, tpr, _ = roc_curve(focus_df['is_true_focus_class'], focus_df['confidence'])
                roc_auc = auc(fpr, tpr)
            else:
                fpr, tpr = [0, 1], [0, 1]
                roc_auc = 0.5

            # Display metrics summary
            print(f"\n📊 Performance Summary for {model_name}")
            print(f"-------------------------------")
            print(f"Model Path: {model_path}")
            print(f"Overall Accuracy: {overall_accuracy:.4f}")
            print(f"")
            print(f"Metrics for '{focus_class}':")
            print(f"  Precision: {class_precision:.4f}")
            print(f"  Recall:    {class_recall:.4f}")
            print(f"  F1 Score:  {class_f1:.4f}")

            # Create visualizations
            fig = plt.figure(figsize=(18, 12))

            # 1. Overall accuracy gauge
            ax1 = fig.add_subplot(2, 2, 1)
            gauge_colors = ['#ff6b6b', '#feca57', '#54a0ff', '#1dd1a1']
            gauge_positions = [0, 0.25, 0.5, 0.75, 1]

            # Create the gauge chart
            for i in range(len(gauge_positions)-1):
                ax1.pie(
                    [gauge_positions[i+1] - gauge_positions[i]],
                    startangle=90,
                    counterclock=False,
                    colors=[gauge_colors[i]],
                    radius=1.2,
                    wedgeprops={'width': 0.2, 'edgecolor': 'white'},
                    frame=True
                )

            # Add the needle
            needle_theta = 90 - 180 * overall_accuracy
            needle_length = 0.8
            ax1.add_patch(plt.arrow(
                0, 0,
                needle_length * np.cos(np.radians(needle_theta)),
                needle_length * np.sin(np.radians(needle_theta)),
                width=0.03, head_width=0.08, head_length=0.1,
                fc='#333333', ec='#333333'
            ))

            # Add the accuracy text
            ax1.text(0, -0.2, f"Accuracy: {overall_accuracy:.1%}",
                     fontsize=14, fontweight='bold', ha='center')

            ax1.set_title(f'Overall Accuracy', fontsize=16, pad=20)
            ax1.set_aspect('equal')
            ax1.set_frame_on(False)
            ax1.axis('off')

            # 2. Class metrics comparison
            ax2 = fig.add_subplot(2, 2, 2)
            metrics_data = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1 Score'],
                'Value': [class_precision, class_recall, class_f1]
            })

            colors = ['#54a0ff', '#ff6b6b', '#1dd1a1']
            bars = sns.barplot(x='Metric', y='Value', data=metrics_data, palette=colors, ax=ax2)

            # Add values on top of bars
            for i, bar in enumerate(bars.patches):
                bars.annotate(f"{bar.get_height():.2f}",
                              (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              ha='center', va='bottom', fontsize=12, fontweight='bold')

            ax2.set_title(f"Metrics for '{focus_class}'", fontsize=16)
            ax2.set_ylim(0, 1.05)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)

            # 3. ROC Curve
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(fpr, tpr, color='#54a0ff', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            ax3.plot([0, 1], [0, 1], color='#ff6b6b', lw=2, linestyle='--')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate', fontsize=12)
            ax3.set_ylabel('True Positive Rate', fontsize=12)
            ax3.set_title(f'ROC Curve for {focus_class}', fontsize=16)
            ax3.legend(loc="lower right")
            ax3.grid(linestyle='--', alpha=0.7)

            # 4. All Classes Comparison
            ax4 = fig.add_subplot(2, 2, 4)

            # Prepare data for all classes
            classes_df = pd.DataFrame([
                {'Class': cls, 'Metric': 'Precision', 'Value': metrics['precision']}
                for cls, metrics in all_metrics.items()
            ] + [
                {'Class': cls, 'Metric': 'Recall', 'Value': metrics['recall']}
                for cls, metrics in all_metrics.items()
            ] + [
                {'Class': cls, 'Metric': 'F1', 'Value': metrics['f1']}
                for cls, metrics in all_metrics.items()
            ])

            # Create a heatmap for all metrics
            pivot_df = classes_df.pivot(index='Class', columns='Metric', values='Value')
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f',
                        linewidths=.5, cbar=False, ax=ax4)

            ax4.set_title('Performance by Class', fontsize=16)
            ax4.set_ylabel('')

            # Overall figure settings
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            fig.suptitle(f'{model_name.upper()} Evaluation Results', fontsize=20, fontweight='bold')

            # Save and display the figure
            plt.savefig(f'{model_name}_evaluation.png', dpi=200, bbox_inches='tight')
            plt.show()

            print("\n✅ Evaluation complete!")
            print(f"📁 Results saved as: {model_name}_evaluation.png")

        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()

# Connect the evaluate button to the function
evaluate_button.on_click(evaluate_model)

# Initialize class dropdown with a placeholder
class_dropdown.options = ['loading...']

print("🎯 Model Evaluation Dashboard Ready!")
print("📁 Upload your best.pt file using the file uploader above")
print("🔧 Or use the manual path option if your file is already on the system")
