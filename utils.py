import numpy as np
import cv2
from sklearn.metrics import jaccard_score
import base64

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (256, 256))
    return img / 255.0

def predict_mask(model, img_arr):
    img_arr = np.expand_dims(img_arr, axis=0)
    pred_mask = model.predict(img_arr)[0]
    return (pred_mask > 0.5).astype(np.uint8)

def compute_change(mask1, mask2):
    return np.abs(mask2 - mask1)

def jaccard_index(mask1, mask2):
    return jaccard_score(mask1.flatten(), mask2.flatten())

def get_download_link(array, filename):
    _, buffer = cv2.imencode(".png", array * 255)
    b64 = base64.b64encode(buffer).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>'
