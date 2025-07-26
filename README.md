# ML-Assisted Image Compression with SVD and CNN

This project implements a machine learning-driven image compression pipeline combining Singular Value Decomposition (SVD) and a Convolutional Neural Network (CNN). The system predicts the optimal SVD compression rank (`k`) for each image to balance perceptual quality (SSIM, PSNR) and file size reduction. A Streamlit web app is included for real-time image compression and metric visualization.

---

## 🚀 Features

- 📷 Compress images using SVD with adaptive rank prediction.
- 🤖 CNN model trained to predict optimal rank `k` from 64×64 grayscale patches.
- 🌈 Uses YUV color space prioritizing luminance (Y) over chrominance (U/V).
- 📉 File size reduced up to 50% using `zlib` after compression.
- 📊 Metrics: SSIM, PSNR, File size comparison with JPEG and PNG.
- 🧪 Robust model training using:
  - Data augmentation (rotation, flipping, brightness)
  - L2 regularization
  - K-fold cross-validation
- 🌐 Deployed as a web app using Streamlit Cloud.

---

## 🧠 Technical Details

### Dataset
- **Source**: BSDS500
- **Input Shape**: Images resized to 256×256 and normalized to [0, 1]

### Preprocessing
- CNN input: Grayscale 64×64 images
- Optional features:
  - Mean, Std
  - Edge density
  - Entropy
  - GLCM contrast
  - Sobel gradients

---

### 🧮 SVD Compression
- Applied to RGB or YUV channels
- Retains top-`k` singular values
- **YUV Mode**:
  - High `k` for Y (luminance)
  - Lower `k` for U and V

---

### 🕸 CNN Architecture

| Layer            | Details                            |
|------------------|-------------------------------------|
| Input            | 64×64×1 grayscale image             |
| Conv2D + ReLU    | 32 filters                          |
| Conv2D + ReLU    | 64 filters                          |
| Conv2D + ReLU    | 128 filters                         |
| MaxPooling2D     | 2×2                                 |
| Flatten + Dense  | 128 → 64 units                      |
| Dropout          | 0.3                                 |
| L2 Regularization| λ = 0.001                           |
| Output           | Sigmoid → Scaled to [10, 256]       |
| Loss             | MSE                                 |
| Optimizer        | Adam                                |

---

### 📊 Quality Metric for Training
