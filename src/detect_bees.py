import cv2
import torch
import os
import numpy as np

# --- Settings ---
# Change to the directory where YOLOv5 is located
# Example: os.chdir(r"C:\Users\your_user\documents\yolov5")
os.chdir(r"C:\Users\jhenr\Downloads\github_bee_project\yolov5")

# Input video path
video_path = r"C:\Users\jhenr\Downloads\github_bee_project\examples\bee_test_video.mp4"

# Path to save output video
output_path = r"C:\Users\jhenr\Downloads\github_bee_project\results\Video_With_Detections_and_Counting.mp4"

# Path to your trained model's weights,
model_weights_path = r"C:\Users\jhenr\Downloads\github_bee_project\models\best.pt"
# Minimum confidence to consider a detection valid (0.0 to 1.0)
conf_thres = 0.25
# --- END OF SETTINGS ---

try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Verify that the path to YOLOv5 and model weights are correct.")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

bar_height = 60
output_height = frame_height + bar_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, output_height))

print("Processing video... Press 'q' in the preview window to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or reading error.")
        break

    results = model(frame)
    
   
    detections = results.xyxy[0].cpu().numpy()

    valid_detections = [det for det in detections if det[4] >= conf_thres]

    num_bees = len(valid_detections)

    for det in valid_detections:
        x1, y1, x2, y2, conf, cls = det
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f'Bee: {conf:.2f}'
        
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    info_bar = np.zeros((bar_height, frame_width, 3), dtype=np.uint8)
    
    info_text = f"Bee Counting on the Board: {num_bees}"
    
    cv2.putText(info_bar, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    combined_frame = np.vstack((frame, info_bar))

    out.write(combined_frame)
    
    cv2.imshow("Bees Detection", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Finalizing and saving the video...")
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved in: {output_path}")
