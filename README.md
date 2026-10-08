# YOLO 2026 — Object Detection

A YOLO-based object detection application built with Python and Streamlit.

Image → YOLO Model → Object Detection → Bounding Boxes

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

The application uses the pretrained YOLO model for object detection.

The Streamlit application provides a web interface for uploading images and detecting objects.

The deployment environment needs:

* streamlit
* ultralytics
* opencv-python-headless
* pillow
