# 🧠 ML-Assisted Image Compression with SVD and CNN

This project implements a machine learning-driven image compression pipeline that combines **Singular Value Decomposition (SVD)** and a **Convolutional Neural Network (CNN)**. The CNN predicts the optimal compression rank `k`, optimizing a balance between perceptual image quality and storage size. A **Streamlit web app** provides real-time compression, metric comparison, and download functionality.

---

## 🚀 Features

- 📷 Compress images using **SVD with adaptive rank prediction**
- 🤖 **CNN model** trained on grayscale 64×64 patches to predict rank `k`
- 🌈 Leverages **YUV color space** — prioritizing luminance (Y channel)
- 📦 File size reduction by up to **50% using `zlib`**
- 📊 Compare **SSIM, PSNR**, and size across ML-SVD, JPEG, and PNG
- 🧪 Robust model training with:
  - Data augmentation (rotation, flipping, brightness)
  - L2 regularization
  - K-fold cross-validation
- 🌐 Deployed on **Streamlit Cloud** for browser-based interaction

---

## 🧠 Technical Details

### 📁 Dataset

- **Source**: BSDS500
- **Image Shape**: Resized to 256×256 and normalized to [0, 1]

### ⚙️ Preprocessing

- **CNN Input**: Grayscale 64×64 image patches
- **Optional handcrafted features**:
  - Mean, Standard Deviation
  - Edge density
  - Entropy
  - GLCM contrast
  - Sobel gradients

---

## 🧮 SVD Compression

- Applied to **RGB** or **YUV** color spaces
- Retains top-`k` singular values for each channel
- **YUV mode**:
  - Higher `k` for **Y channel (luminance)**
  - Lower `k` for **U/V channels (chrominance)**

---

## 🧱 CNN Architecture

| Layer            | Configuration                        |
|------------------|--------------------------------------|
| Input            | 64×64×1 grayscale image              |
| Conv2D + ReLU    | 32 filters                           |
| Conv2D + ReLU    | 64 filters                           |
| Conv2D + ReLU    | 128 filters                          |
| MaxPooling2D     | 2×2                                  |
| Flatten + Dense  | 128 → 64 units                       |
| Dropout          | 0.3                                  |
| Regularization   | L2 (λ = 0.001)                       |
| Output           | Sigmoid activation, scaled to [10, 256] |
| Loss             | Mean Squared Error (MSE)             |
| Optimizer        | Adam                                 |

---

### 🧪 Training Objective Function

```python
quality_score = 0.8 * SSIM + 0.4 * (PSNR / 40) - 0.05 * (Size / 240)


## 📊 Evaluation Results

| Format         | SSIM    | PSNR (dB) | File Size     |
|----------------|---------|-----------|---------------|
| ML-SVD (base)  | 0.6327  | 24.03     | 57.11 KB      |
| ML-SVD (CNN)   | ~0.85   | ~30       | ~20–30 KB     |
| JPEG           | 0.8382  | 26.43     | 11.21 KB      |
| PNG            | 0.9997  | 53.94     | 133.37 KB     |

---

## 📁 Project Structure
ml-svd-compression/
├── model/
│ ├── ml_svd_model.keras # Trained CNN model
│ ├── X_mean.npy # Normalization mean
│ ├── X_std.npy # Normalization std
├── step6_cnn.py # CNN training script
├── step7_evaluate_ml_svd.py # Evaluate ML-SVD
├── step8_benchmark.py # Benchmark vs JPEG/PNG
├── streamlit_app.py # Streamlit web app
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🛠 Installation & Usage

### 🔧 Clone and Install

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
pip install -r requirements.txt

▶️ Run Locally
streamlit run streamlit_app.py

