from ultralytics import YOLO
import cv2

model = YOLO("models/best2.pt")

results = model(
    "videos/person_knife.mp4",
    stream=True,
    conf=0.25
)

for result in results:

    frame = result.plot()

   
    detected_classes = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        detected_classes.append(class_name)

    if "knife" in detected_classes:
        print("🚨 ALERT: A person is detected with a knife in CAM 1!")

    cv2.imshow("YOLO Detection - CAM 1", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# from ultralytics import YOLO

# model = YOLO("models/best2.pt")

# results = model(
#     "test/differentKnifeImages.png",
#     conf=0.2
# )

# results[0].show()