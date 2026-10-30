# from ultralytics import YOLO
# import cv2

# model = YOLO("models/yolo26n.pt")

# # cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("YOLO Tracking", 1280, 720)

# # # Default tracker (TrackTrack)
# # results = model.track(source="videos/input_video.mp4", show=True)

# # Switch to ByteTrack
# results = model.track(source="videos/input_video.mp4", show=True, tracker="bytetrack.yaml")

from ultralytics import YOLO

# Load an official or custom model
# model = YOLO("models/yolo26n.pt")  # Load an official Detect model
model = YOLO("yolo26n-seg.pt")  # Load an official Segment model
model = YOLO("yolo26n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom-trained model

# Perform tracking with the model
#results = model.track("videos/input_video.mp4", show=True)  # Tracking with default tracker
results = model.track("videos/input_video.mp4", show=True, tracker="bytetrack.yaml")  # with ByteTrack