import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="YOLO Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Detect Objects"):
        model = load_model()

        with st.spinner("Detecting objects..."):
            results = model.predict(
                image,
                imgsz=320,
                device="cpu",
                verbose=False
            )

        result_image = results[0].plot()
        st.image(result_image, caption="Detection Result", use_container_width=True)

        for box in results[0].boxes:
            name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            st.write(f"{name} — {confidence:.2f}")
