# Save this as app.py

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Lung Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Lung Cancer Detection using Vision Transformer")
st.write("Upload a histopathology image to predict the tissue class.")

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    import gdown
    import os

    model_path = "vit_lung_model_80_20.pth"
    file_id = "1Dc4bGiKwU6Uur4yRDJ4aAHk4DPOOQQe-"

    # Always ensure a clean file (prevents corrupted/HTML downloads)
    if os.path.exists(model_path):
        os.remove(model_path)

    # Download using file_id (more reliable than URL)
    gdown.download(id=file_id, output=model_path, quiet=False)

    # Safety check
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 5_000_000:
        st.error("Model download failed or file is corrupted.")
        st.stop()

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

with st.spinner("Loading model..."):
    model = load_model()
# ---------------------------
# Classes
# ---------------------------
class_names = {
    0: "Lung Adenocarcinoma",
    1: "Normal Lung Tissue",
    2: "Lung Squamous Cell Carcinoma"
}

# ---------------------------
# Image Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img_tensor).logits
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence = conf.item() * 100

    st.subheader("Predicted Tissue Class")
    st.success(predicted_class)
    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))

# ---------------------------
# Footer
# ---------------------------
st.caption("AI-based tissue classification for educational/research use only. Not a medical diagnosis.")
