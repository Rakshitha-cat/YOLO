import os
import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered"
)

st.title("YOLO Object Detection")
st.write("Upload an image to detect objects.")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "yolov8n.onnx"
)

IMG_SIZE = 320
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


@st.cache_resource
def load_model():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )

    return ort.InferenceSession(
        MODEL_PATH,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )


def detect_objects(image, session):
    original = np.array(image.convert("RGB"))
    original_height, original_width = original.shape[:2]

    scale = min(
        IMG_SIZE / original_width,
        IMG_SIZE / original_height
    )

    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized = cv2.resize(
        original,
        (new_width, new_height)
    )

    canvas = np.full(
        (IMG_SIZE, IMG_SIZE, 3),
        114,
        dtype=np.uint8
    )

    pad_x = (IMG_SIZE - new_width) // 2
    pad_y = (IMG_SIZE - new_height) // 2

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    input_data = canvas.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)

    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {input_name: input_data}
    )[0]

    predictions = output[0]

    if predictions.shape[0] == 84:
        predictions = predictions.T

    boxes = []
    scores = []
    class_ids = []

    for prediction in predictions:
        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < CONF_THRESHOLD:
            continue

        center_x, center_y, box_width, box_height = prediction[:4]

        x1 = (center_x - box_width / 2 - pad_x) / scale
        y1 = (center_y - box_height / 2 - pad_y) / scale
        x2 = (center_x + box_width / 2 - pad_x) / scale
        y2 = (center_y + box_height / 2 - pad_y) / scale

        x1 = max(0, min(int(x1), original_width - 1))
        y1 = max(0, min(int(y1), original_height - 1))
        x2 = max(0, min(int(x2), original_width - 1))
        y2 = max(0, min(int(y2), original_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    result_image = original.copy()

    if boxes:
        selected = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            CONF_THRESHOLD,
            IOU_THRESHOLD
        )

        for index in np.array(selected).flatten():
            x, y, width, height = boxes[int(index)]
            class_id = class_ids[int(index)]
            confidence = scores[int(index)]

            name = CLASS_NAMES[class_id]

            cv2.rectangle(
                result_image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            label = f"{name} {confidence:.0%}"

            cv2.putText(
                result_image,
                label,
                (x, max(y - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        count = len(selected)
    else:
        count = 0

    return result_image, count


if not os.path.isfile(MODEL_PATH):
    st.error(
        "Model not found. Ensure yolov8n.onnx is in the "
        "same project directory as app.py."
    )
    st.stop()


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Original Image",
            width="stretch"
        )

        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                session = load_model()
                result_image, count = detect_objects(
                    image,
                    session
                )

            st.subheader("Detection Results")

            st.image(
                result_image,
                caption="Detected Objects",
                width="stretch"
            )

            st.success(f"Detected {count} object(s).")

    except Exception as error:
        st.error(f"Detection failed: {error}")
