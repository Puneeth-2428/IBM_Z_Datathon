from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'dyslexia_detector.h5')
VAL_DIR = os.path.join(os.path.dirname(__file__), '..', 'image_classification', 'val')

model = load_model(MODEL_PATH)
print('Loaded model from', MODEL_PATH)

classes = sorted([d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])
print('Detected classes (sorted):', classes)

y_true = []
y_pred = []
filenames = []
for i, cls in enumerate(classes):
    folder = os.path.join(VAL_DIR, cls)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f'Processing class {cls}: found {len(files)} files')
    for fn in files:
        p = os.path.join(folder, fn)
        img = Image.open(p).convert('RGB').resize((224,224))
        arr = np.expand_dims(np.array(img)/255.0, axis=0)
        pred = model.predict(arr)
        score = float(pred[0][0]) if np.array(pred).ndim == 2 else float(pred[0])
        # Using threshold 0.5
        label_idx = int(score > 0.5)
        y_true.append(i)
        y_pred.append(label_idx)
        filenames.append(fn)

print('\nProcessed samples per class:')
from collections import Counter
print(Counter(y_true))

if len(set(y_true)) < len(classes):
    print('\nWarning: some classes have 0 samples in validation set; classification report may fail.')

if len(y_true) > 0 and len(set(y_true)) == len(classes):
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=classes))
    print('\nConfusion matrix:')
    print(confusion_matrix(y_true, y_pred))
else:
    print('\nSkipping classification report due to missing classes or no samples.')
