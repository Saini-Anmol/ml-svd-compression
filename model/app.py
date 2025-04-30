import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import os
import tensorflow as tf
import logging
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom CSS for vibrant, eye-catching UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    .main {background: linear-gradient(135deg, #e0f7fa 0%, #80deea 100%);}
    .stButton>button {
        background: linear-gradient(45deg, #0288d1, #4fc3f7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 20px;
        font-family: 'Poppins', sans-serif;
        font-size: 16px;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stFileUploader {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #01579b;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
        color: #0288d1;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #4fc3f7;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .welcome-banner {
        background: linear-gradient(45deg, #0288d1, #4fc3f7);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and normalization parameters
model_path = "model/ml_svd_model.keras"
X_mean_path = "model/X_mean.npy"
X_std_path = "model/X_std.npy"

try:
    model = tf.keras.models.load_model(model_path)
    X_mean = np.load(X_mean_path)
    X_std = np.load(X_std_path)
except Exception as e:
    st.error(f"Error loading model or parameters: {e}")
    st.stop()

# SVD compression function with quantization
def svd_compress(image, k):
    """
    Compress an RGB image using SVD with rank k truncation and 8-bit quantization.
    
    Args:
        image: NumPy array of shape (256, 256, 3), values in [0, 1]
        k: Number of singular values to retain
    
    Returns:
        Compressed image as NumPy array
    """
    img = image.astype(np.float32)
    compressed_img = np.zeros_like(img)
    scaling_factors = []  # Store max singular values for decompression
    
    for channel in range(3):
        # SVD decomposition
        U, S, Vt = np.linalg.svd(img[:, :, channel], full_matrices=False)
        
        # Quantize singular values (S) to 8-bit
        S_max = np.max(S[:k])
        scaling_factors.append(S_max)
        S_quantized = np.round((S[:k] / S_max) * 255).astype(np.uint8)  # Scale to [0, 255]
        
        # Quantize U and Vt to 8-bit (values in [-1, 1] scaled to [0, 255])
        U_quantized = np.round((U[:, :k] + 1) * 127.5).astype(np.uint8)  # [-1, 1] -> [0, 255]
        Vt_quantized = np.round((Vt[:k, :] + 1) * 127.5).astype(np.uint8)
        
        # Decompress: Rescale back to original ranges
        S_dequantized = (S_quantized / 255.0) * S_max
        U_dequantized = (U_quantized / 127.5) - 1
        Vt_dequantized = (Vt_quantized / 127.5) - 1
        
        # Reconstruct channel
        compressed_img[:, :, channel] = np.dot(U_dequantized * S_dequantized, Vt_dequantized)
    
    compressed_img = np.clip(compressed_img, 0, 1)
    return compressed_img

# Compute SSIM and PSNR metrics
def compute_metrics(original, compressed):
    if original.shape != compressed.shape or original.shape != (256, 256, 3):
        raise ValueError(f"Expected images of shape (256, 256, 3), got original: {original.shape}, compressed: {compressed.shape}")
    original = np.clip(original, 0, 1)
    compressed = np.clip(compressed, 0, 1)
    ssim_score = ssim(original, compressed, channel_axis=2, data_range=1.0, win_size=7)
    mse = np.mean((original - compressed) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    return ssim_score, psnr

# Extract features for ML model
def extract_features(image):
    try:
        gray_img = np.mean(image, axis=2)
        edge_density = np.mean(canny(gray_img, sigma=1.0))
        hist, _ = np.histogram(image, bins=256, range=(0, 1))
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) / hist.size
        features = [np.mean(image), np.std(image), edge_density, entropy]
        return np.array(features)
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise

# Estimate ML-SVD file size with 8-bit quantization
def estimate_ml_svd_size(k, height=256, width=256, bytes_per_element=1):
    elements_per_channel = k * (height + width + 1)
    total_elements = 3 * elements_per_channel
    size_bytes = total_elements * bytes_per_element
    return size_bytes / 1024

# Estimate original image size
def estimate_original_size(height=256, width=256, channels=3, bytes_per_pixel=1):
    size_bytes = height * width * channels * bytes_per_pixel
    return size_bytes / 1024

# JPEG compression
def compress_jpeg(image, quality=50):
    img = (image * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    size_kb = len(buffer) / 1024
    compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR) / 255.0
    return compressed, size_kb

# PNG compression
def compress_png(image):
    img = (image * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', img)
    size_kb = len(buffer) / 1024
    compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR) / 255.0
    return compressed, size_kb

# ML-SVD compression
def ml_svd_compress(image, model, X_mean, X_std, max_k=256):
    features = extract_features(image)
    features_norm = (features - X_mean) / X_std
    k_pred = model.predict(np.array([features_norm]), verbose=0)[0][0] * max_k
    k_pred = int(np.clip(k_pred, 10, max_k))
    compressed_img = svd_compress(image, k_pred)
    return compressed_img, k_pred

# Welcome banner
st.markdown("""
<div class='welcome-banner'>
    <h1>ML-SVD Image Compression 
BY- Anmol Saini</h1>
    <p>Transform your images with cutting-edge ML-based compression. Upload a JPG to see the magic!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional controls and info
with st.sidebar:
    st.header("Controls & Info")
    st.markdown("<p style='font-family: Poppins, sans-serif; color: #01579b;'>Upload your image to start!</p>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a JPG image", type=["jpg", "jpeg"])
    
    if uploaded_file:
        # Thumbnail preview
        thumbnail = Image.open(uploaded_file).resize((100, 100))
        st.image(thumbnail, caption="Preview")
    
    # Info section
    with st.expander("🔍 Learn More"):
        st.markdown("""
        **What is ML-SVD?**  
        Uses machine learning to optimize Singular Value Decomposition for image compression.  
        **SSIM**: Measures structural similarity (0-1, higher is better).  
        **PSNR**: Measures signal-to-noise ratio (dB, higher is better).  
        **Compressed Size**: Estimated size using 8-bit quantization.
        """)

# Main content with tabs
if uploaded_file:
    with st.spinner("🔄 Compressing your image..."):
        try:
            # Read and preprocess image
            image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(image) / 255.0
            img_resized = cv2.resize(img_array, (256, 256))

            # Compress using ML-SVD
            compressed_ml, k_pred = ml_svd_compress(img_resized, model, X_mean, X_std)
            ssim_ml, psnr_ml = compute_metrics(img_resized, compressed_ml)
            size_ml = estimate_ml_svd_size(k_pred)
            original_size = estimate_original_size()

            # Compress using JPEG and PNG
            compressed_jpeg, size_jpeg = compress_jpeg(img_resized)
            ssim_jpeg, psnr_jpeg = compute_metrics(img_resized, compressed_jpeg)
            compressed_png, size_png = compress_png(img_resized)
            ssim_png, psnr_png = compute_metrics(img_resized, compressed_png)

            # Tabs for organized content
            tab1, tab2, tab3 = st.tabs(["📷 Original", "🖼️ Compressed", "📊 Metrics"])

            with tab1:
                st.header("Original Image")
                st.image(image, caption="Your Uploaded Image", use_column_width=True)

            with tab2:
                st.header("Compressed Image (ML-SVD)")
                st.image(compressed_ml, caption=f"Compressed with k={k_pred}", use_column_width=True)
                
                # Download button
                compressed_pil = Image.fromarray((compressed_ml * 255).astype(np.uint8))
                buf = io.BytesIO()
                compressed_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="⬇️ Download Compressed Image",
                    data=byte_im,
                    file_name="compressed_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            with tab3:
                st.header("Compression Metrics")
                st.markdown("<p style='font-family: Poppins, sans-serif; color: #01579b;'>Compare ML-SVD with JPEG and PNG</p>", unsafe_allow_html=True)
                
                # Metrics in cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Method", "ML-SVD")
                    st.metric("SSIM", f"{ssim_ml:.4f}")
                    st.metric("PSNR (dB)", f"{psnr_ml:.2f}")
                    st.metric("Compressed Size (KB)", f"{size_ml:.2f}")
                    st.metric("Original Size (KB)", f"{original_size:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Method", "JPEG")
                    st.metric("SSIM", f"{ssim_jpeg:.4f}")
                    st.metric("PSNR (dB)", f"{psnr_jpeg:.2f}")
                    st.metric("Compressed Size (KB)", f"{size_jpeg:.2f}")
                    st.metric("Original Size (KB)", f"{original_size:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Method", "PNG")
                    st.metric("SSIM", f"{ssim_png:.4f}")
                    st.metric("PSNR (dB)", f"{psnr_png:.2f}")
                    st.metric("Compressed Size (KB)", f"{size_png:.2f}")
                    st.metric("Original Size (KB)", f"{original_size:.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Side-by-side comparison
                st.header("Side-by-Side Comparison")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.image(image, caption="Original Image", use_column_width=True)
                with col_right:
                    st.image(compressed_ml, caption="Compressed (ML-SVD)", use_column_width=True)

        except Exception as e:
            st.error(f"🚨 Error processing image: {e}")

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; background: #ffffff; border-radius: 10px; margin-top: 20px;'>
    <p style='font-family: Poppins, sans-serif; color: #01579b;'>
        Built with Streamlit for ML-SVD Image Compression Project by Anmol Saini
    </p>
</div>
""", unsafe_allow_html=True)
