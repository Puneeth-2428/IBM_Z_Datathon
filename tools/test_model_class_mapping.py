from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'dyslexia_detector.h5')
TRAIN_LABEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'image_classification', 'train')

model = load_model(MODEL_PATH)
print('Loaded model from', MODEL_PATH)

classes = sorted([d for d in os.listdir(TRAIN_LABEL_DIR) if os.path.isdir(os.path.join(TRAIN_LABEL_DIR, d))])
print('Detected classes (sorted):', classes)

for cls in classes:
    folder = os.path.join(TRAIN_LABEL_DIR, cls)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]
    scores = []
    for fn in files:
        p = os.path.join(folder, fn)
        img = Image.open(p).convert('RGB').resize((224,224))
        arr = np.expand_dims(np.array(img)/255.0, axis=0)
        pred = model.predict(arr)
        score = float(pred[0][0]) if np.array(pred).ndim == 2 else float(pred[0])
        scores.append(score)
    if len(scores) == 0:
        print(cls, '-> no images')
        continue
    avg = sum(scores)/len(scores)
    frac_pos = sum(1 for s in scores if s>0.5) / len(scores)
    print(f"Class: {cls} | samples: {len(scores)} | avg_score: {avg:.4f} | frac_score>0.5: {frac_pos:.2f}")
