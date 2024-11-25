import cv2
from ultralytics import YOLO

import numpy as np
import cv2
from mss import mss
from PIL import Image

def main() :
    bounding_box = {'top': 500, 'left': 450, 'width': 500, 'height': 500}
    sct = mss()
    # Load the YOLO model
    model = YOLO("best.pt")
    model.model.names = {0: 'c'}
    # Open the video file
    # video_path = "path/to/your/video/file.mp4"
    # cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while True:
        # Read a frame from the video
        frame = sct.grab(bounding_box)

        # Convert the frame to OpenCV format
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGRA2BGR)
    
        # Run YOLO inference on the frame
        results = model(frame,imgsz=640,conf=0.17,iou=0.1,device=[0],augment=True)
        #augment : improving detection robustness at the cost of inference speed.
        #iou : to not get multiple detection for single crack
    
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    
if __name__=="__main__" :
    main()