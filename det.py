from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")

results = model("videos/person_knife.mp4", stream=True,conf=0.5)

for result in results:

    # Draw bounding boxes on the current frame
    frame = result.plot()

    # Display current frame
    cv2.imshow("YOLO Detection", frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
   


cv2.destroyAllWindows()