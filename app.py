import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import matplotlib
import requests

# ---- Hugging Face model config ----
HF_REPO = "DanLDevs/malaria-cell-detection"

# ---- Model Download & Load ----
MODEL_FILES = {
  "custom_cnn_weights_only.weights.h5": f"https://huggingface.co/{HF_REPO}/resolve/main/custom_cnn_weights_only.weights.h5",
  "mobilenetv2_frozen_weights_only.weights.h5": f"https://huggingface.co/{HF_REPO}/resolve/main/malaria_mobilenetv2_frozen_weights_only.weights.h5",
  "mobilenetv2_finetuned_weights_only.weights.h5": f"https://huggingface.co/{HF_REPO}/resolve/main/malaria_mobilenetv2_finetuned_weights_only.weights.h5"
}

MODEL_DIR = "models"

def download_models():
  os.makedirs(MODEL_DIR, exist_ok=True)

  for filename, file_id in MODEL_FILES.items():
    output_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(output_path):
      response = requests.get(file_id, stream=True)
      response.raise_for_status()
      with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)

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
MOBILENET_LAST_CONV = "Conv_1"
# Helper function to get last Conv2D layer name
def get_last_conv_layer_name(model):
  for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
      return layer.name
  raise ValueError("No Conv2D layer found in model.")

# Grad-CAM heatmap generator
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
  """
  Generate Grad-CAM heatmap.
  pred_index: 0 = Infected (low sigmoid), 1 = Uninfected (high sigmoid).
  If None, uses the predicted class automatically.
  """
  grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[
      model.get_layer(last_conv_layer_name).output,
      model.output
    ],
  )

  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array, training=False)
    # Auto-select class based on prediction if not specified
    if pred_index is None:
      pred_index = int(predictions[0][0] >= 0.5)
    loss = predictions[:, 0] if pred_index == 1 else (1 - predictions[:, 0])
  
  grads = tape.gradient(loss, conv_outputs)

  # Pool gradients over spatial dims
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
  conv_outputs = conv_outputs[0]
  heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  # ReLU + Normalize
  heatmap = tf.maximum(heatmap, 0)
  heatmap /= (tf.reduce_max(heatmap) + 1e-8)

  return heatmap.numpy()

# Overlay display helper (returns image)
def overlay_gradcam_full(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
  """
  Standard addition Grad-CAM overlay. Returns a PIL Image.
  alpha controls heatmap intensity (lower = more subtle).
  """
  # Convert heatmap to 0-1
  heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
  heatmap_pil = Image.fromarray(heatmap_uint8, mode="L").resize(img.size, Image.BILINEAR)
  heatmap_np = np.array(heatmap_pil) / 255.0

  colormap = matplotlib.colormaps["jet"] # No deprecation warning
  heatmap_colored = np.uint8(255 * colormap(heatmap_np)[:, :, :3])

  img_np = np.array(img).astype(np.float32)
  overlay = (1 - alpha) * img_np + alpha * heatmap_colored.astype(np.float32)
  return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))

# Preprocess
def preprocess_for_model(img: Image.Image, model_choice: str) -> np.ndarray:
  img = img.resize((128,128))
  arr = np.array(img).astype(np.float32)

  if "MobileNetV2" in model_choice:
    arr = mobilenet_preprocess(arr)
  else:
    arr = arr / 255.0

  return np.expand_dims(arr, axis=0)

# ---- Load Models (cached) ----
@st.cache_resource
def load_all_models():
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

# ---- Streamlit UI ----
st.title("Malaria Cell Detection App")
st.info(
  "Upload a microscopic cell image to detect malaria infection. "
  "Enable Grad-CAM to visualize which regions drove the prediction. "
  "Want to try more images? Download the dataset from "
  "[Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)"
)

# Load models
with st.spinner("Loading models (first run may take a minute)..."):
  models_dict = load_all_models()

show_gradcam = st.checkbox("Show Grad-CAM")
gradcam_model_name = None
if show_gradcam:
  gradcam_model_name = st.selectbox(
    "Model for Grad-CAM:",
    list(models_dict.keys())
  )

# Sample images
st.subheader("Don't have an image? Try a sample!")
col_s1, col_s2 = st.columns(2)
uploaded_image = None

with col_s1:
  if st.button("Use Infected Sample"):
    uploaded_image = Image.open("samples/infected_sample.png").convert("RGB")
  
with col_s2:
  if st.button("Use Uninfected Sample"):
    uploaded_image = Image.open("samples/uninfected_sample.png").convert("RGB")

# File uploader
file = st.file_uploader("Upload a Cell Image", type=['jpg', 'png', 'jpeg'])
if file:
  uploaded_image = Image.open(file).convert("RGB")

# Run predictions for both sample and uploaded image
if uploaded_image is not None:
  col1, col2 = st.columns(2)
  with col1:
    st.image(uploaded_image, caption="Uploaded Image", width=400)
  # Run all models
  results = {}
  for name, model_obj in models_dict.items():
    img_array = preprocess_for_model(uploaded_image, name)
    pred = float(model_obj(img_array, training=False)[0][0])
    if pred >= 0.5:
      pred_class, confidence = "Uninfected", pred
    else:
      pred_class, confidence = "Infected", 1 - pred
    results[name] = {"class": pred_class, "confidence": confidence}
    
  st.subheader("Model Predictions")
  cols = st.columns(len(results))
  for col, (name, res) in zip(cols, results.items()):
    with col:
      color = "✅" if res["class"] == "Uninfected" else "❌"
      st.metric(label=name, value=f"{color} {res['class']}", delta=f"{res['confidence']:.1%} confidence")
  # Grad-CAM
  if show_gradcam and gradcam_model_name:
    with st.spinner("Generating Grad-CAM..."):
      try:
        best_model_obj = models_dict[gradcam_model_name]
        last_conv = MOBILENET_LAST_CONV if "MobileNetV2" in gradcam_model_name else get_last_conv_layer_name(best_model_obj)
        img_array = preprocess_for_model(uploaded_image, gradcam_model_name)
        heatmap = make_gradcam_heatmap(img_array, best_model_obj, last_conv)
        overlay = overlay_gradcam_full(uploaded_image, heatmap, alpha=0.45)
        with col2:
          st.image(overlay, caption=f"Grad-CAM ({gradcam_model_name})", width=400)
      except Exception  as e:
        st.error(f"Grad-CAM failed: {e}")