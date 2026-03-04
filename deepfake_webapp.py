# deepfake_webapp.py
# =====================================================
# 🖼️ Deepfake Detection Web App
# =====================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# -----------------------
# 1️⃣ Device setup
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# 2️⃣ Load model
# -----------------------
@st.cache_resource
def load_model():
    # Recreate EfficientNet-B0 architecture
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 2)
    )
    # Load your trained state_dict
    state_dict = torch.load("deepfake_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------
# 3️⃣ Load class names
# -----------------------
@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# -----------------------
# 4️⃣ Image preprocessing
# -----------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# -----------------------
# 5️⃣ Streamlit UI
# -----------------------
st.title("🖼️ Deepfake Detection Web App")
st.write("Upload an image to check if it’s REAL or FAKE.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
        prediction = class_names[str(pred_class)]

    st.success(f"Prediction: **{prediction}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    # Optional: show bar chart for both classes
    st.bar_chart({class_names["0"]: probs[0].item(), class_names["1"]: probs[1].item()})