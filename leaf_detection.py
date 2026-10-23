from ultralytics import YOLO

# Load a pretrained segmentation model
model = YOLO("runs/segment/train-7/weights/best.pt")

# Analyze your photograph
results = model.predict(
    source="leaf2.jpg",
    save=True,
    conf=0.25
)

print("Detection complete!")