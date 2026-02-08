import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Upload Photo + Predict", page_icon="😷")

st.title("😷 Face Mask Detection")
st.write("Take a photo or upload one, then run it through the CNN model and optionally save it")

# ---------- Settings ----------
MODEL_PATH = "Face_Mask_Detection.keras"
IMG_SIZE = (64, 64)  # must match training target_size
THRESHOLD = 0.5

# Folder where images will be saved (on the computer/server)
SAVE_DIR = Path("saved_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

def save_uploaded_image(file_obj, prefix: str = "img") -> Path:
    original_name = getattr(file_obj, "name", "") or ""
    ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique}{ext}"
    out_path = SAVE_DIR / filename

    data = file_obj.getvalue()
    out_path.write_bytes(data)
    return out_path

def preprocess_for_model(uploaded_file) -> np.ndarray:
    """
    Converts Streamlit uploaded image to model-ready numpy array:
    shape: (1, 64, 64, 3), dtype float32, scaled to [0,1]
    """
    # Streamlit UploadedFile -> bytes -> PIL Image
    img = Image.open(uploaded_file)

    # Ensure 3 channels (RGB). This also handles PNG with alpha (RGBA)
    img = img.convert("RGB")

    # Resize to model expected input
    img = img.resize(IMG_SIZE)

    # PIL -> np array
    arr = np.array(img, dtype=np.float32)

    # Normalize
    arr /= 255.0

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)  # (1, 64, 64, 3)
    return arr

# ---------- UI ----------
st.subheader("1) Take a picture (phone camera)")
camera_photo = st.camera_input("Open camera and take a photo")

st.subheader("2) Or upload from gallery")
gallery_photo = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

chosen = camera_photo if camera_photo is not None else gallery_photo

if chosen is not None:
    st.image(chosen, caption="Preview", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("😷 Predict face mask"):
            try:
                x = preprocess_for_model(chosen)
                pred = model.predict(x, verbose=0)[0]  # Get all 3 class probabilities
                st.write(f"Model output: **{pred}**")

                # 3-class prediction: 0=without_mask, 1=with_mask, 2=incorrect_mask
                class_idx = np.argmax(pred)
                confidence = pred[class_idx]

                if class_idx == 0:
                    label = "Incorrect Mask ⚠️"
                elif class_idx == 1:
                    label = "With Mask ✅"
                else:
                    label = "Without Mask ❌"

                st.success(f"Prediction: **{label}** ({confidence:.2%} confidence)")

                # Show probabilities for all classes
                st.info(f"Without Mask: {pred[0]:.2%} | With Mask: {pred[1]:.2%} | Incorrect Mask: {pred[2]:.2%}")

                # Debug: show the shape to prove it's correct
                st.caption(f"Sent to model with shape: {x.shape} (should be (1, 64, 64, 3))")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        if st.button("💾 Save image to computer"):
            try:
                saved_path = save_uploaded_image(chosen, prefix="phone")
                st.success(f"Saved ✅  {saved_path.resolve()}")
                st.info("The image was saved on the computer running Streamlit (server)")
            except Exception as e:
                st.error(f"Failed to save: {e}")

st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")
st.caption(f"Model file: {Path(MODEL_PATH).resolve()}")