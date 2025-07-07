import os
import numpy as np
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_unet

IMAGE_DIR = "data/glacial/images"
MASK_DIR = "data/glacial/masks"
MODEL_PATH = "models/unet_glacial.h5"
IMG_SIZE = 256

def load_data(img_dir, mask_dir):
    X, Y = [], []
    for filename in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, filename))
        mask = cv2.imread(os.path.join(mask_dir, filename), 0)
        if img is not None and mask is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
            X.append(img)
            Y.append(mask.reshape(IMG_SIZE, IMG_SIZE, 1))
    return np.array(X), np.array(Y)

X, Y = load_data(IMAGE_DIR, MASK_DIR)
model = build_unet()
model.fit(X, Y, validation_split=0.1, batch_size=8, epochs=5,
          callbacks=[ModelCheckpoint(MODEL_PATH, save_best_only=True)])
