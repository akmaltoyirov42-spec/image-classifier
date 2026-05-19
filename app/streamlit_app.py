import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from PIL import Image

from src.predict import predict_image

st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="centered")

st.title("🔍 Image Classifier")
st.write("Upload an image and the model will tell you what it sees. Built with EfficientNet-B0 fine-tuned on a custom dataset.")

uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Your image", use_column_width=True)

    with st.spinner("Running inference..."):
        try:
            result = predict_image(image)
        except FileNotFoundError:
            st.error("No trained model found. Run `python -m src.train` first.")
            st.stop()

    st.markdown(f"### Prediction: **{result['prediction']}**")
    st.markdown(f"Confidence: `{result['confidence']*100:.1f}%`")

    st.divider()
    st.markdown("**Top 5 predictions:**")
    for item in result["top5"]:
        bar_val = item["confidence"]
        st.markdown(f"`{item['label']}`")
        st.progress(bar_val, text=f"{bar_val*100:.1f}%")
