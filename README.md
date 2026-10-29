# YOLO Object Detector (Webcam + Image Upload)

A beginner-friendly local Python application that **uses YOLO26n for real object detection**.

## Features

- **Upload Image**: select JPG, PNG, BMP, or WebP and see YOLO detection boxes.
- **Start Webcam / Stop Webcam**: run YOLO on live camera frames.
- **Confidence slider**: adjust the minimum confidence required to show an object.
- **Save Result**: export the latest annotated image or webcam frame.
- **Object count**: see how many objects have been detected and the most common classes.

## Requirements

- Python 3.11 or 3.12 recommended
- macOS, Windows, or Linux with Tkinter available
- Internet connection on the **first run only**, so the pretrained `yolo26n.pt` weights can download
- A webcam for the webcam feature (not required for image upload)

## Setup — macOS / Linux

```bash
cd YOLO_Object_Detector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If `python3 -m tkinter` fails on macOS, install a Python distribution that includes Tkinter (for example, from python.org). Some Homebrew installations need the matching `python-tk` package.

## Setup — Windows PowerShell

```powershell
cd YOLO_Object_Detector
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If activation is blocked, you can run `.venv\Scripts\python.exe -m pip install -r requirements.txt` then `.venv\Scripts\python.exe app.py` instead.

## How it works

1. OpenCV reads the webcam frame or Pillow loads an uploaded image.
2. `YOLO("yolo26n.pt")` loads the pretrained COCO object detection model.
3. `model.predict(...)` produces boxes, class IDs, and confidence scores.
4. Ultralytics draws annotated detection boxes on the image.
5. Tkinter displays the image with labels, and lets you save the result.

**Note:** This model detects common object categories (person, cup, phone, chair, etc.). It does **not** recognize facial expressions, identify specific people, or detect arbitrary objects outside its trained categories.

## Tips

- If the webcam does not open, allow camera access in macOS System Settings > Privacy & Security > Camera and close competing video apps.
- For faster processing, keep lighting good and reduce other computer workloads.
- To try a different pretrained model, change `MODEL_NAME` near the top of `app.py`.
- First model inference can take longer than subsequent ones.

Official documentation: https://docs.ultralytics.com/models/yolo26/
