import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("YOLO Object Detection")
st.write("Upload an image to detect objects using YOLO.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Original Image",
        width="stretch"
    )

    if st.button("Detect Objects"):
        try:
            with st.spinner("Detecting objects..."):
                model = load_model()

                results = model.predict(
                    image,
                    imgsz=320,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

                detected_image = result.plot()
                detected_image = detected_image[:, :, ::-1]

                st.subheader("Detection Results")
                st.image(
                    detected_image,
                    caption="Detected Objects",
                    width="stretch"
                )

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    st.success(
                        f"Detected {len(boxes)} object(s)."
                    )

                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        name = result.names[class_id]

                        st.write(
                            f"**{name}** — "
                            f"Confidence: {confidence:.2%}"
                        )
                else:
                    st.info("No objects detected.")

        except Exception as error:
            st.error(f"Detection failed: {error}")

