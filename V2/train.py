from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("last.pt")
    # Train the model
    results = model.train(data="data_v2.yaml", epochs=15, imgsz=640, workers=1, device=[0])
    
    
if __name__ == "__main__":
    main()