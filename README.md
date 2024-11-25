
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

Run crack detection on video
```bash
  cd V2
  python test_on_vid.py
```
+ I dont have any specific video to thats why I am using small part of screen as Input for our model (we can decide what to display on that part of screen our that part of screen to varify working of model)


