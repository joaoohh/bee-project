# 🐝 Bee Detection with YOLOv5

A college project for automated bee detection and counting in videos using YOLOv5. This system helps monitor bee activity by detecting and counting bees in video footage.

**Note**: This is an academic project and is still in development.

## 🚀 Features

- **Real-time bee detection** using custom-trained YOLOv5 model
- **Automatic counting** with frame-by-frame statistics
- **Video processing** with visual bounding boxes and confidence scores
- **Progress tracking** and detailed processing information
- **Customizable confidence threshold** for detection accuracy

## 📋 Requirements

- Python 3.7+
- PyTorch
- OpenCV
- YOLOv5

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/bee-detection-yolov5.git
cd bee-detection-yolov5
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install YOLOv5**
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

## 📁 Project Structure

```
bee-detection-yolov5/
├── src/
│   └── detect_bees.py          # Main detection script
├── models/
│   └── best.pt                 # Custom trained model weights
├── examples/
│   └── bee_test_video.mp4      # Sample input video
├── results/
│   └── output_videos/          # Processed videos with detections
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🎯 Usage

### Configuration
Edit the settings at the top of `src/detect_bees.py`:

```python
# Input video path
video_path = r"examples/bee_test_video.mp4"

# Output video path
output_path = r"results/output_video.mp4"

# Model weights path
model_weights_path = 'models/best.pt'

# Confidence threshold (0.0 to 1.0)
conf_thres = 0.25
```

### Run Detection
```bash
python src/detect_bees.py
```

### Example Output
The processed video will include:
- Green bounding boxes around detected bees
- Confidence scores for each detection
- Real-time frame counter and bee count
- Progress bar at the bottom

## 📊 Model Information

This project uses a custom-trained YOLOv5 model specifically optimized for bee detection:

- **Input size**: 640x640 pixels
- **Classes**: 1 (bee)
- **Training framework**: YOLOv5
- **Confidence threshold**: 0.25 (adjustable)

## 🔧 Customization

### Adjusting Detection Sensitivity
Modify the `conf_thres` value in the script:
- Lower values (0.1-0.3): More sensitive, may include false positives
- Higher values (0.4-0.8): Less sensitive, more accurate detections

### Training Your Own Model
If you want to train your own bee detection model:
1. Prepare your dataset with bee images and YOLO format labels
2. Follow the YOLOv5 training guide
3. Replace `models/best.pt` with your trained weights

## 🙏 Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework