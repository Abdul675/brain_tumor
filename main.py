import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# -----------------------------
# Load your trained CNN model
# -----------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("best_resnet_mri.h5")  # update path
    return model

model = load_cnn_model()

# Define class labels (change as per your training)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


IMG_SIZE = (224, 224)

# -----------------------------
# Grad-CAM function

def get_gradcam_heatmap(img_array, model, class_idx=None, layer_name=None):
    # Step 1: get prediction
    preds = model(img_array)
    if class_idx is None:
        class_idx = tf.argmax(preds[0]).numpy().item()  # ✅ safe conversion
    else:
        class_idx = int(class_idx)

    # Step 2: pick last conv layer if none provided
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Step 3: compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # 🟢 FIX FOR GRADIENTS: Watch the conv layer output
        tape.watch(conv_outputs)
        
        if isinstance(predictions, list):
            predictions = predictions[0]
            
        # ✅ FIX FOR SLICING: Use tf.gather
        loss = tf.gather(predictions, class_idx, axis=1) 

    grads = tape.gradient(loss, conv_outputs)

    # The error now is fixed because grads is no longer None
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

    return heatmap, class_idx

def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlayed

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🧠 Tumor Detection with CNN", layout="centered")

st.title("🧠 Brain Tumor Detection using CNN")
st.markdown("""
This AI-powered app detects **brain tumor types** from MRI scans  
and highlights the **focus area** using **Grad-CAM heatmaps**.  

### Features:
- 📂 Upload your MRI image  
- 🧾 Predict tumor type (Glioma, Meningioma, Pituitary, No Tumor)  
- 🔥 Visualize focus regions (Grad-CAM)  
- 📊 Display prediction probabilities  
""")

uploaded_file = st.file_uploader("📂 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    
    # Read & prepr
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

# Apply ResNet preprocessing instead of manual /255
    input_arr = img_to_array(img_resized)
    input_arr = preprocess_input(input_arr)   # ✅ ensures [-1, 1] scaling
    input_arr = np.expand_dims(input_arr, axis=0)


    # Model prediction
    preds = model.predict(input_arr)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx] * 100

    st.subheader("✅ Prediction Results")
    st.write(f"**Predicted Tumor Type:** {CLASS_NAMES[class_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show prediction probabilities
    st.bar_chart(preds[0])
    
    
    print("model.input_shape:", model.input_shape)
    print("model.output_shape:", model.output_shape)
    print("Available conv-like layers (sample):", [l.name for l in model.layers if 'conv' in l.name][-10:])
    preds = model.predict(input_arr)
    print("preds shape:", preds.shape, "preds:", preds)
    print("predicted class (python int):", int(np.argmax(preds[0])))


    # Grad-CAM
    heatmap, class_idx = get_gradcam_heatmap(input_arr, model)
    overlayed_img = overlay_heatmap(img_rgb, heatmap)

    # Display original and heatmap side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Uploaded MRI Image", use_container_width=True)
    with col2:
        st.image(overlayed_img, caption="Focus Area (Grad-CAM)", use_container_width=True)
