# AntrikshaNetra - Streamlit UI (app.py)

import utils.session
import streamlit as st
from utils import preprocess_image, predict_mask, compute_change, jaccard_index, get_download_link
from model import load_or_build_model
import os
from tensorflow.keras.models import load_model
from model import build_unet
st.set_page_config("AntrikshaNetra", layout="wide")
st.title("🛰️ AntrikshaNetra")
st.markdown("AI-powered feature detection and change analysis using satellite imagery.")

# Sidebar Feature Selection
feature = st.sidebar.selectbox("Select Feature", ["glacial", "road", "drainage"])
model = build_unet()
model = load_model("models/unet_glacial.h5", compile=False)
model.summary()

# Upload Inputs
col1, col2 = st.columns(2)
with col1:
    uploaded_t1 = st.file_uploader("Upload T1 Image", type=["png", "jpg", "jpeg"], key="t1")
with col2:
    uploaded_t2 = st.file_uploader("Upload T2 Image", type=["png", "jpg", "jpeg"], key="t2")

# Run Analysis
if uploaded_t1 and uploaded_t2:
    with st.spinner("Processing images..."):
        img1 = preprocess_image(uploaded_t1)
        img2 = preprocess_image(uploaded_t2)
        mask1 = predict_mask(model, img1)
        mask2 = predict_mask(model, img2)
        change_map = compute_change(mask1, mask2)
        jaccard = jaccard_index(mask1, mask2)

    # Output Display
    st.subheader(f"🧠 Results for {feature.upper()} Detection")
    col1, col2, col3 = st.columns(3)
    col1.image(mask1 * 255, caption="T1 Mask", use_container_width=True, clamp=True)
    col2.image(mask2 * 255, caption="T2 Mask", use_container_width=True, clamp=True)
    col3.image(change_map * 255, caption="Change Map", use_container_width=True, clamp=True)

    st.success(f"✅ Jaccard Similarity: {jaccard:.3f}")

    # Download Options
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    col4.markdown(get_download_link(mask1, f"{feature}_mask_t1.png"), unsafe_allow_html=True)
    col5.markdown(get_download_link(mask2, f"{feature}_mask_t2.png"), unsafe_allow_html=True)
    col6.markdown(get_download_link(change_map, f"{feature}_change_map.png"), unsafe_allow_html=True)
else:
    st.info("Upload both T1 and T2 images to begin analysis.")
