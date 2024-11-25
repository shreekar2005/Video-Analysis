from ultralytics import YOLO

def main():
    
    # Load a pretrained YOLO11n model
    model = YOLO("best.pt")

    # Predict on an image
    model.predict("image.png", show=True, imgsz=640, conf=0.5, device=[0])


if __name__ == "__main__":
    main()