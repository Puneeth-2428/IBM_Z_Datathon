import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

st.title("✍️ Dyslexia Detection from Handwriting")

# Use the models folder next to this script
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# Choose an existing model file from the repo
MODEL_FILENAME = "dyslexia_detector.h5"
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

if not os.path.exists(model_path):
    available = []
    try:
        available = os.listdir(MODEL_DIR)
    except Exception:
        available = []
    st.error(f"Model file not found: {model_path}\nAvailable files in models/: {available}")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Infer classes in the same order image_dataset_from_directory would (sorted folder names)
TRAIN_CLASS_DIR = os.path.join(os.path.dirname(__file__), 'image_classification', 'train')
try:
    classes = sorted([d for d in os.listdir(TRAIN_CLASS_DIR) if os.path.isdir(os.path.join(TRAIN_CLASS_DIR, d))])
    # image_dataset_from_directory maps class names to integer labels by sorted folder name order
    # e.g. classes[0] -> label 0, classes[1] -> label 1
except Exception:
    # Fallback to default if train folder not available
    classes = ['Non-Dyslexic', 'Dyslexic']

st.write(f"Using class order: {classes}")

# Persistent small config to remember which class corresponds to the model's
# positive output (score > threshold). This helps when training label order
# differs from what you expect in the UI.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'app_config.json')
config = {}
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as cf:
            config = json.load(cf)
    except Exception:
        config = {}

uploaded = st.file_uploader("Upload a handwriting image", type=['jpg','png','jpeg'])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)
    img_resized = img.resize((224, 224))
    arr = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    try:
        pred = model.predict(arr)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # If model outputs a single sigmoid value
    score = float(pred[0][0]) if np.array(pred).ndim == 2 else float(pred[0])

    # Allow user to tune threshold for classification clarity
    threshold = st.slider('Decision threshold (lower = more likely positive)', 0.0, 1.0, 0.5, 0.01)

    is_positive = score > threshold

    # Let the user explicitly choose which class corresponds to the model's
    # positive output (score > threshold). This is clearer than a flip checkbox.
    default_positive = config.get('positive_class') if config.get('positive_class') in classes else (classes[1] if len(classes) > 1 else classes[0])
    positive_class = st.radio("Model positive corresponds to:", options=classes, index=classes.index(default_positive))

    # Save preference option
    if st.button('Save mapping preference'):
        try:
            with open(CONFIG_PATH, 'w') as cf:
                json.dump({'positive_class': positive_class}, cf)
            st.success(f"Saved mapping: positive -> {positive_class}")
        except Exception as e:
            st.error(f"Failed to save mapping: {e}")

    # Compute the predicted label index from the boolean is_positive and
    # the chosen positive class index.
    pos_idx = classes.index(positive_class)
    other_idx = 1 - pos_idx if len(classes) > 1 else 0
    idx = pos_idx if is_positive else other_idx
    label = classes[idx]

    # Display a clear badge and probability
    st.subheader(f"Prediction: {label}")
    st.metric(label="Confidence (sigmoid score)", value=f"{score:.4f}")

    # Helpful textual guidance: determine if the predicted label corresponds to 'Dyslexic'
    dyslexic_index = None
    try:
        dyslexic_index = classes.index('Dyslexic')
    except ValueError:
        dyslexic_index = None

    if dyslexic_index is not None:
        if idx == dyslexic_index:
            st.info("Model indicates dyslexia-like handwriting. This is a prediction — not a diagnosis.")
        else:
            st.success("Model indicates non-dyslexic handwriting characteristics.")
    else:
        # If we don't have a 'Dyslexic' label in classes, fall back to keyword check in label text
        if 'dyslex' in label.lower():
            st.info("Model indicates dyslexia-like handwriting. This is a prediction — not a diagnosis.")
        else:
            st.success("Model indicates non-dyslexic handwriting characteristics.")

    # Show raw prediction array for debugging/clarity
    with st.expander('Show raw model output'):
        st.write(pred)
