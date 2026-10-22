from ultralytics import YOLO

# Load a pretrained segmentation model
model = YOLO("yolo11n-seg.pt")

# Train YOLO on the leaf dataset
model.train(
    data="leaf_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)

print("Training complete!")
