import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍"
)

st.title("🔍 YOLO Object Detection")
st.write("Upload an image and YOLO will detect objects in it.")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects"):

        results = model(image)

        result_image = results[0].plot()

        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

        st.subheader("Detected Objects")

        boxes = results[0].boxes

        if len(boxes) == 0:
            st.write("No objects detected.")
        else:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                object_name = model.names[class_id]

                st.write(
                    f"**{object_name}** - "
                    f"Confidence: {confidence:.2f}"
                )