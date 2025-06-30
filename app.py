import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from models.efficientnet_transformer import StrokeClassifier
import yaml
import os
from datetime import datetime

# === Load Config ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Load Model ===
@st.cache_resource
def load_model():
    model = StrokeClassifier(
        backbone=config['model']['backbone'],
        pretrained=False,
        num_classes=config['model']['num_classes']
    )
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# === Session state for history ===
if 'history' not in st.session_state:
    st.session_state.history = []

# === UI Layout ===
st.set_page_config(page_title="Stroke Classifier", layout="wide")
st.title("🧠 Stroke Detection from CT Scans")

with st.sidebar:
    st.header("📚 About This App")
    st.markdown("""
    This app uses a deep learning model (EfficientNet-B0) to classify brain CT scans as:
    - 🟢 **No Stroke** (0)
    - 🔴 **Stroke** (1)

    **Business Context:**
    > This tool supports clinicians and hospitals by providing fast, AI-assisted stroke triage in emergency and radiology settings.

    Upload a scan to see predictions, confidence levels, and model behavior.
    """)

# === Upload Interface ===
uploaded_file = st.file_uploader("📂 Upload a CT scan (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # === Preprocess ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    input_tensor = transform(image).unsqueeze(0)

    # === Predict ===
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label_map = {0: "🟢 No Stroke", 1: "🔴 Stroke"}
    result = label_map[pred]
    confidence = float(probs[0][pred])

    # === Display Result ===
    st.markdown("### 🧠 Prediction Result")
    st.success(f"{result}  —  Confidence: {confidence:.2%}")

    st.progress(confidence)

    # === Update History ===
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "Time": timestamp,
        "Filename": uploaded_file.name,
        "Prediction": result,
        "Confidence": f"{confidence:.3f}"
    })

# === History Table ===
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📊 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download History as CSV", csv, "stroke_predictions.csv", "text/csv")
