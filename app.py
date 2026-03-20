import os, gdown, zipfile
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ---- Model Download & Load ----
MODEL_FILES = {
  "custom_cnn_weights_only.weights.h5": "18wJ49TfpXiZksOOLSrfveyWws_VtEw9y",
  "mobilenetv2_frozen_weights_only.weights.h5": "1rwTZRkq5gOqwajVOdinK6E58ALGBQ4uZ",
  "mobilenetv2_finetuned_weights_only.weights.h5": "1DrvxWVJi3pYaZrkDNkZd0kreqIdurj0-"
}

MODEL_DIR = "models"

def download_models():
  os.makedirs(MODEL_DIR, exist_ok=True)

  for filename, file_id in MODEL_FILES.items():
    output_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(output_path):
      url = f"https://drive.google.com/uc?id={file_id}"
      print(f"Downloading {filename}...")
      gdown.download(url, output_path, quiet=True)

# ---- Model Builders ----
# Custom CNN Architecture
def build_custom_cnn(input_shape=(128, 128,3)):
  inputs = tf.keras.Input(shape=input_shape)

  x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)

  x = tf.keras.layers.Conv2D(64, (3,3), activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(2,2)(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.5)(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  return model

# MobileNetV2 (Frozen)
def build_mobilenetv2_frozen():
  base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
  )

  base_model.trainable = False

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.5)(x)
  output = Dense(1, activation="sigmoid")(x)
  model = Model(inputs=base_model.input, outputs=output)
  return model

# MobileNetV2 (Fine-Tuned)
def build_mobilenetv2_finetuned(input_shape=(128, 128,3)):
  base_model = MobileNetV2(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet"
  )

  # Unfreeze last 30 layers
  for layer in base_model.layers[:-30]:
    layer.trainable = False
  for layer in base_model.layers[-30:]:
    layer.trainable = True

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.5)(x)
  output = Dense(1, activation="sigmoid")(x)
  
  model = Model(inputs=base_model.input, outputs=output)

  return model

# ---- Helper Functions ----
# Helper function to get last Conv2D layer name
def get_last_conv_layer_name(model):
  for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
      return layer.name
  raise ValueError("No Conv2D layer found in model.")

# Grad-CAM heatmap generator
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
  # Create model that outputs both the conv layer and predictions
  grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
      model.get_layer(last_conv_layer_name).output,
      model.output
    ],
  )

  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array, training=False)
    class_index = 0
    loss = predictions[:, class_index]
  
  grads = tape.gradient(loss, conv_outputs)

  # Global average pooling of gradients
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  # Normalize
  heatmap = tf.maximum(heatmap, 0)
  heatmap /= (tf.reduce_max(heatmap) + 1e-8)

  return heatmap.numpy()

# Overlay display helper (returns image)
def overlay_gradcam_full(img, heatmap, alpha=0.4):
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# Preprocess
def preprocess_for_model(img, model_choice):
  img = img.resize((128,128))
  arr = np.array(img)

  if "MobileNetV2" in model_choice:
    arr = mobilenet_preprocess(arr)
  else:
    arr = arr / 255.0

  return np.expand_dims(arr, axis=0)

# ---- Load Models (cached) ----
# Cache models so they are loaded only once
@st.cache_resource
def load_models():
  download_models()
  
  # --- Custom CNN ---
  custom_model = build_custom_cnn()
  custom_model.load_weights(
    os.path.join(MODEL_DIR, "custom_cnn_weights_only.weights.h5")
  )

  # --- MobileNetV2 Frozen ---
  mobilenet_frozen = build_mobilenetv2_frozen()
  mobilenet_frozen.load_weights(
    os.path.join(MODEL_DIR, "mobilenetv2_frozen_weights_only.weights.h5")
  )

  # --- MobileNetV2 Fine-Tuned ---
  mobilenet_finetuned = build_mobilenetv2_finetuned()
  mobilenet_finetuned.load_weights(
    os.path.join(MODEL_DIR, "mobilenetv2_finetuned_weights_only.weights.h5")
  )

  return {
    "Custom CNN": custom_model,
    "MobileNetV2 (Frozen)": mobilenet_frozen,
    "MobileNetV2 (Fine-tuned)": mobilenet_finetuned
  }

# Load models
with st.spinner("Loading models..."):
  models_dict = load_models()

# ---- Streamlit UI ----
st.title("Malaria Cell Detection App")
st.info(
  "This model analyzes microscopic cell images to detect malaria infection. "
  "Grad-CAM highlights regions most important for the prediction."
)

show_gradcam = st.checkbox("Show Grad-CAM")

gradcam_model_name = None
if show_gradcam:
  gradcam_model_name = st.selectbox(
    "Select model for Grad-CAM:",
    list(models_dict.keys())
  )

# File uploader
file = st.file_uploader("Upload a Cell Image", type=['jpg', 'png', 'jpeg'])

if file:

  col1, col2 = st.columns(2)
  uploaded_image = Image.open(file).convert("RGB")
  with col1:
    st.image(uploaded_image, caption="Uploaded Image", width=400)

  # Dictionary to store predictions
  results = {}

  # Run through all models
  for name, model_obj in models_dict.items():
    img_array = preprocess_for_model(uploaded_image, name)
    pred = float(model_obj(img_array, training=False)[0][0])

    # Calculate confidence
    threshold = 0.5

    if pred >= threshold:
      pred_class = "Uninfected"
      confidence = pred
    else:
      pred_class = "Infected"
      confidence = 1 - pred
    results[name] = {"class": pred_class, "confidence": confidence}
    
  st.subheader("Model Prediction Summary")

  # Show all model predictions
  st.subheader("All Model Predictions")
  cols = st.columns(len(results))
  for col, (name, res) in zip(cols, results.items()):
    with col:
      st.write(f"**{name}**")
      st.write(f"{res['class']}")
      st.write(f"{res['confidence']:.2%}")

  # Grad-CAM
  if show_gradcam:
    try:
      best_model_obj = models_dict[gradcam_model_name]
      last_conv_layer_name = get_last_conv_layer_name(best_model_obj)
      img_array = preprocess_for_model(uploaded_image, gradcam_model_name)
      heatmap = make_gradcam_heatmap(img_array, best_model_obj, last_conv_layer_name)

      img_array_np = np.array(uploaded_image)
      heatmap_resized = cv2.resize(heatmap, (img_array_np.shape[1], img_array_np.shape[0]))
      overlay = overlay_gradcam_full(img_array_np, heatmap_resized, alpha=0.4)

      with col2:
        st.image(overlay, caption=f"Grad-CAM ({gradcam_model_name})", width=400)

    except Exception  as e:
      st.error(f"Grad-CAM failed: {e}")
