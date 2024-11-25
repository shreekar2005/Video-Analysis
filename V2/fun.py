from ultralytics import YOLO

def main():
    model=YOLO("best.pt")
    model.names=["crack"]
    print(model.names)
    
if __name__ == "__main__":
    main()