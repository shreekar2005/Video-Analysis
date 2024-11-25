from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("best.pt")

    # Validate the model
    metrics = model.val(workers=1, split="test", device=[0])
    print(metrics.box.map)  # map50-95  

if __name__ == "__main__":
    main()