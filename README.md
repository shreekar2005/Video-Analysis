
# video Analysis

This project uses a machine learning model to detect cracks in images and videos, with scripts for training and testing to ensure accurate and efficient detection. There are two folders, V1 and V2, which have the same functionality, but in V2, the model is trained on a dataset with a different split. You can use either, though V2 is recommended.



## Run Locally

Clone the project

```bash
  git clone https://github.com/shreekar2005/Video-Analysis.git
```

Go to the project directory

```bash
  cd Video-Analysis
```

Activate your virtual environment and Install dependencies

```bash
  pip install ultralytics
  pip install opencv-python
  pip install mss
  pip install pillow
```

To train/ run on GPU Download and Install following dependencies : 

1.CudaToolkit (https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)

2.CudaLibrary (https://pytorch.org/get-started/locally/)

follow this video for better understanding : https://youtu.be/cL05xtTocmY?feature=shared

Run crack detection on video:
```bash
  cd V2
  python test_on_vid.py
```
+ I dont have any specific video to thats why I am using small part of screen as Input for our model (we can decide what to display on that part of screen our that part of screen to varify working of model)

## video
![](https://github.com/shreekar2005/Video-Analysis/blob/main/dc_vid.gif)


## About data and model
Data split
train : 3222
val : 281
test : 522 (using for validate model after training)

...trained for 75 epochs (using train and val data)

mAP50-95 : 0.5608 (after validating on test data)

## More about dataset and YoloV11
Dataset link : https://universe.roboflow.com/iitj-gxmjr/crack-detection-qlrkm

YoloV11 : https://docs.ultralytics.com/models/yolo11/#usage-examples

how to use YoloV11 locally : https://docs.ultralytics.com/modes/train/

