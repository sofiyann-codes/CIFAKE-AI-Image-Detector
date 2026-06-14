import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="CIFAKE - AI Image Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Model Definition ─────────────────────────────
class CIFAKE_CNN(nn.Module):
    def __init__(self):
        super(CIFAKE_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ── Grad-CAM ─────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.conv_layers[10]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor):
        self.model.zero_grad()
        output = self.model(img_tensor)
        output.backward()
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])
        activations  = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]
        heatmap = activations.mean(dim=0).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
        return heatmap

# ── Load Model ────────────────────────────────────
@st.cache_resource
def load_model():
    model = CIFAKE_CNN()
    model.load_state_dict(torch.load(
        '../models/cifake_cnn.pth',
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model

# ── Transform ─────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ── UI ────────────────────────────────────────────
st.title("🔍 CIFAKE — AI Image Detector")
st.markdown("Upload any image to detect if it's **REAL** or **AI-Generated (FAKE)**")
st.markdown("---")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        prob = model(img_tensor).item()

    pred  = 'REAL' if prob >= 0.5 else 'FAKE'
    conf  = prob if prob >= 0.5 else 1 - prob

    # Grad-CAM
    gcam    = GradCAM(model)
    img_t   = transform(image).unsqueeze(0)
    heatmap = gcam.generate(img_t)

    img_np          = np.array(image.resize((32, 32))) / 255.0
    heatmap_resized = cv2.resize(heatmap, (32, 32))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay         = np.clip(img_np * 0.6 + heatmap_colored/255.0 * 0.4, 0, 1)

    with col2:
        st.subheader("Heatmap")
        st.image(heatmap_resized, clamp=True, use_column_width=True)

    with col3:
        st.subheader("Overlay")
        st.image(overlay, clamp=True, use_column_width=True)

    st.markdown("---")

    # Result
    if pred == 'REAL':
        st.success(f"✅ This image is REAL — Confidence: {conf:.1%}")
    else:
        st.error(f"🚨 This image is AI-GENERATED (FAKE) — Confidence: {conf:.1%}")

    st.progress(float(conf))
    st.markdown(f"**Prediction score:** {prob:.4f} (above 0.5 = REAL, below 0.5 = FAKE)")