from ultralytics import YOLO

# Load a pretrained segmentation model
model = YOLO("yolo26n-seg.pt")

# Analyze your photograph
results = model.predict(
    source="leaf1.jpg",
    save=True,
    conf=0.25
)

print("Detection complete!")