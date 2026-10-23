
from ultralytics import YOLO


def main():
    # Load a pretrained segmentation model
    model = YOLO("yolo11n-seg.pt")

    # Train YOLO on the leaf dataset
    model.train(
        data="leaf_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
