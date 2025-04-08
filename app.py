import streamlit as st
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from streamlit_image_comparison import image_comparison
import io

# === Page Config ===
st.set_page_config(page_title="Image Lab ✨", layout="wide")

# === Styling ===
st.markdown("""
<style>
h1 {
    text-align: center;
    font-size: 3rem;
    background: linear-gradient(to right, #4B8BBE, #306998);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5em;
}
img {
    border-radius: 12px;
}
.section {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("<h1>🖼️ Image Processing Lab</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: gray;'>Upload an image, select operations, and process beautifully.</p>", unsafe_allow_html=True)

# === Split Layout ===
left_col, right_col = st.columns([1, 1.2])

with left_col:
    uploaded_file = st.file_uploader("📄 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert("RGB"))
        processed_img = img_array.copy()

        if "selected_ops" not in st.session_state:
            st.session_state.selected_ops = []

        def handle_conflicts(selected):
            if "Grayscale" in selected:
                selected = [op for op in selected if op not in ["Brightness/Contrast", "Invert Colors"]]
            if "Histogram Equalization" in selected:
                selected = [op for op in selected if op != "Invert Colors"]
            return selected

        st.markdown("## 🧰 Choose & Order Operations")
        ordered_ops = st.multiselect("Select operations to apply (in order):", [
            "Resize", "Grayscale", "Smoothing", "Sharpening", "Invert Colors", "Brightness/Contrast",
            "Edge Detection", "Histogram Equalization", "Rotate", "Flip", "Add Text"
        ], default=handle_conflicts(st.session_state.selected_ops))
        st.session_state.selected_ops = ordered_ops

        st.markdown("## ⚙️ Operation Settings")
        with st.expander("Adjust Parameters", expanded=True):
            width, height = img_array.shape[1], img_array.shape[0]
            kernel_size = 5
            brightness, contrast = 1.0, 0
            threshold1, threshold2 = 50, 150
            rotate_deg = 0
            flip_mode = 'None'
            text_input = ""
            text_position = (50, 50)
            font_scale = 1
            sharpening_strength = 1.5

            if "Resize" in ordered_ops:
                st.subheader("✂️ Resize")
                col1, col2 = st.columns(2)
                width = col1.slider("Width", 50, 1000, width)
                height = col2.slider("Height", 50, 1000, height)

            if "Smoothing" in ordered_ops:
                st.subheader("🩹 Smoothing")
                kernel_size = st.slider("Gaussian Kernel Size (odd)", 1, 15, 5, step=2)

            if "Sharpening" in ordered_ops:
                st.subheader("✨ Sharpening")
                sharpening_strength = st.slider("Sharpening Strength", 1.0, 5.0, 1.5, step=0.1)

            if "Brightness/Contrast" in ordered_ops:
                st.subheader("💡 Brightness & Contrast")
                brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
                contrast = st.slider("Contrast", -100, 100, 0)

            if "Edge Detection" in ordered_ops:
                st.subheader("🔍 Edge Detection")
                threshold1 = st.slider("Threshold1", 0, 255, 50)
                threshold2 = st.slider("Threshold2", 0, 255, 150)

            if "Rotate" in ordered_ops:
                st.subheader("🔄 Rotate")
                rotate_deg = st.slider("Rotate (degrees)", -180, 180, 0)

            if "Flip" in ordered_ops:
                st.subheader("🔁 Flip")
                flip_mode = st.selectbox("Flip Mode", ["None", "Horizontal", "Vertical"])

            if "Add Text" in ordered_ops:
                st.subheader("📝 Add Text")
                text_input = st.text_input("Enter text to add", "Image Lab")
                x = st.slider("Text X", 0, processed_img.shape[1], 50)
                y = st.slider("Text Y", 0, processed_img.shape[0], 50)
                font_scale = st.slider("Font Scale", 0.5, 3.0, 1.0, 0.1)
                text_position = (x, y)

with right_col:
    if uploaded_file:
        for op in ordered_ops:
            if op == "Resize":
                processed_img = cv2.resize(processed_img, (width, height))
            elif op == "Grayscale":
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            elif op == "Smoothing":
                processed_img = cv2.GaussianBlur(processed_img, (kernel_size, kernel_size), 0)
            elif op == "Sharpening":
                kernel = np.array([[0, -1, 0],
                                   [-1, 5 * sharpening_strength, -1],
                                   [0, -1, 0]])
                processed_img = cv2.filter2D(processed_img, -1, kernel)
            elif op == "Invert Colors":
                processed_img = cv2.bitwise_not(processed_img)
            elif op == "Brightness/Contrast":
                processed_img = cv2.convertScaleAbs(processed_img, alpha=brightness, beta=contrast)
            elif op == "Edge Detection":
                if len(processed_img.shape) == 3:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                processed_img = cv2.Canny(processed_img, threshold1, threshold2)
            elif op == "Histogram Equalization":
                if len(processed_img.shape) == 3:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                processed_img = cv2.equalizeHist(processed_img)
            elif op == "Rotate":
                center = (processed_img.shape[1] // 2, processed_img.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
                processed_img = cv2.warpAffine(processed_img, matrix, (processed_img.shape[1], processed_img.shape[0]))
            elif op == "Flip":
                if flip_mode == "Horizontal":
                    processed_img = cv2.flip(processed_img, 1)
                elif flip_mode == "Vertical":
                    processed_img = cv2.flip(processed_img, 0)
            elif op == "Add Text":
                if len(processed_img.shape) == 2:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                cv2.putText(processed_img, text_input, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 0, 0), 2, cv2.LINE_AA)

        st.markdown("## 🔍 Compare")
        image_comparison(
            img1=Image.fromarray(img_array),
            img2=Image.fromarray(processed_img if len(processed_img.shape) == 3 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)),
            label1="Original",
            label2="Processed",
            width=600
        )

        if "Grayscale" in ordered_ops or "Histogram Equalization" in ordered_ops:
            st.markdown("## 📊 Histogram")
            fig, ax = plt.subplots()
            gray = processed_img if len(processed_img.shape) == 2 else cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            ax.hist(gray.ravel(), bins=256, range=[0, 256], color='gray')
            st.pyplot(fig)

        st.markdown("## 📅 Download")
        img_bytes = cv2.imencode('.png', cv2.cvtColor(processed_img if len(processed_img.shape) == 3 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2BGR))[1].tobytes()
        st.download_button(
            label="💾 Download Processed Image",
            data=img_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )
    else:
        st.info("📸 Upload an image on the left to get started!", icon="📷")