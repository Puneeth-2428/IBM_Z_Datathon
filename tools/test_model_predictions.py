from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'dyslexia_detector.h5')
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'images', 'train')

model = load_model(MODEL_PATH)
print('Loaded model from', MODEL_PATH)

files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith('.png') or f.lower().endswith('.jpg')][:10]
for fn in files:
    p = os.path.join(SAMPLE_DIR, fn)
    img = Image.open(p).convert('RGB').resize((224,224))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(arr)
    score = float(pred[0][0]) if np.array(pred).ndim == 2 else float(pred[0])
    print(fn, '->', score)
