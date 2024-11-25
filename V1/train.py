from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from YAML
    # Train the model
    results = model.train(data="data.yaml", epochs=1, imgsz=640, workers=1, device=[0])
    
if __name__ == "__main__":
    main()