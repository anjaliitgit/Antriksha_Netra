import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)
    bn = Conv2D(64, 3, activation='relu', padding='same')(p2)
    u1 = Conv2DTranspose(32, 2, strides=2, padding='same')(bn)
    u1 = concatenate([u1, c2])
    c3 = Conv2D(32, 3, activation='relu', padding='same')(u1)
    u2 = Conv2DTranspose(16, 2, strides=2, padding='same')(c3)
    u2 = concatenate([u2, c1])
    c4 = Conv2D(16, 3, activation='relu', padding='same')(u2)
    outputs = Conv2D(1, 1, activation='sigmoid')(c4)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_or_build_model(feature):
    model_path = os.path.join(MODEL_DIR, f"unet_{feature}.h5")
    if os.path.exists(model_path):
        return load_model(model_path)
    model = build_unet()
    model.save(model_path)
    return model
