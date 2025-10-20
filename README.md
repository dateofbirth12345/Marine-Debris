Marine Surface Debris Detection System
=====================================

Overview
--------
End-to-end project to detect and localize marine surface debris from images.
Includes preprocessing, training (YOLOv8), evaluation, inference, and a Streamlit UI with map output.

Requirements
-----------
- Python 3.10+
- Windows PowerShell (provided)

Install
-------
```bash
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
```

Dataset
-------
By default the config expects the dataset at:
`C:\\Users\\AlgoQuant-HR-Admin\\Downloads\\5151941`

Dataset must be in YOLO format (images/ and labels/ folders). If your dataset
is not in this structure, adjust `src/preprocess.py` to your layout.

Prepare Splits
--------------
```bash
.venv\\Scripts\\python src/preprocess.py
```
This creates `data/splits/` and a `configs/dataset.yolov8.yaml` file.

Train
-----
```bash
.venv\\Scripts\\python src/train.py
```
Best weights path is saved to `models/best_model.path`.

Evaluate
--------
```bash
.venv\\Scripts\\python src/evaluate.py
```
Prints mAP, mAP50, and recall.

Run UI
------
```bash
.venv\\Scripts\\streamlit run app/streamlit_app.py
```
Upload images to visualize detections and GPS points on the map.

Notes
-----
- EXIF GPS is extracted when present to plot markers on the map.
- Update `configs/config.yaml` to change hyperparameters.




