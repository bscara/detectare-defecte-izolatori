import os
import tempfile

import streamlit as st
from ultralytics import YOLO
from PIL import Image


st.set_page_config(
    page_title="Detectare defecte izolatori",
    page_icon="⚡",
    layout="centered"
)

st.title("Detectare automată a defectelor la izolatori")

st.write(
    "Aplicația permite încărcarea unei imagini și identificarea automată "
    "a stării izolatorului sau a defectului detectat, utilizând modelul YOLOv8 antrenat."
)

MODEL_PATH = "best.pt"

CLASS_NAMES_RO = {
    "Glass_Dirty": "Izolator murdar",
    "broken": "Izolator spart",
    "izolator bun": "Izolator bun",
    "pollution-flashover": "Poluare / flashover"
}


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

confidence = st.slider(
    "Prag de încredere",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

uploaded_image = st.file_uploader(
    "Încarcă o imagine pentru testare",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    st.subheader("Imagine încărcată")
    st.image(image, use_container_width=True)

    if st.button("Rulează detecția"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            image_path = temp_file.name

        results = model.predict(
            source=image_path,
            conf=confidence,
            save=False
        )

        result = results[0]
        annotated_image = result.plot()

        st.subheader("Rezultatul detecției")
        st.image(annotated_image, use_container_width=True)

        if len(result.boxes) == 0:
            st.warning("Nu a fost detectat niciun obiect în imagine.")
        else:
            st.success(f"Au fost detectate {len(result.boxes)} obiecte.")

            st.subheader("Detalii detecții")

            for i, box in enumerate(result.boxes, start=1):
                class_id = int(box.cls[0])
                class_name_original = model.names[class_id]
                class_name = CLASS_NAMES_RO.get(class_name_original, class_name_original)
                conf = float(box.conf[0])

                st.write(
                    f"**Detecția {i}:** {class_name} "
                    f"— scor de încredere: {conf:.2f}"
                )

        os.remove(image_path)
else:
    st.info("Încarcă o imagine pentru a rula detecția.")