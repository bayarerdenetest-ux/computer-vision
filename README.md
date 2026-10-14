# Marmot Detection with YOLO26

Detection of marmot burrows/marmots (label: `dosh`) in drone images, trained with Ultralytics YOLO26 on Google Colab.

## Overview

- **Task:** bounding-box detection, one class (`dosh`)
- **Annotation:** CVAT (three tasks merged into one), exported as XML
- **Data split:** random 80% train / 20% val (`seed=0`), converted from CVAT XML to YOLO format inside the notebook
- **Model:** YOLO26n, `imgsz=1280`, 100 epochs, batch 8, Tesla T4 GPU
- **Second experiment:** custom `yolo26n-p234` (P2 head added, P5 head removed) initialized from `yolo26n.pt`

## Repository structure

```
computer-vision/
├── README.md
├── images/
│   ├── prediction_example.png
│   └── f1_curve.png
└── notebooks/
    └── marmot_yolo26_setup.ipynb
```

## How to run

1. Open `notebooks/marmot_yolo26_setup.ipynb` in Google Colab and switch the runtime to GPU.
2. Put `merged_marmot.zip` (images) and `merged_fixed.xml` (CVAT annotations) in Google Drive. The dataset is not included in this repository.
3. Run the cells from top to bottom: install Ultralytics, convert annotations, train, predict.

## Results

F1 reaches its maximum of about 0.53 at a confidence threshold of 0.208, so predictions are shown with `conf=0.2`.

<img width="2250" height="1500" alt="dosh_f1_curve" src="https://github.com/user-attachments/assets/749669c2-1b2f-4b1c-aca7-d990094b8230" />


Example prediction on a validation image:

<img width="4608" height="3456" alt="dosh_zurag" src="https://github.com/user-attachments/assets/69af78d0-b869-4162-b0bd-79fa51762d4f" />


## Known limitations

- Confidence scores are low (mostly 0.2 to 0.4), so a low threshold is needed and overlapping boxes appear in dense areas.
- The model sometimes mistakes dried dung (argal) on the ground for `dosh` in images that contain none.

## Tools

Python, Ultralytics YOLO26, CVAT, Google Colab
