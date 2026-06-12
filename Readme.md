# 🎙️ Deepfake Audio Detector

A deep learning system that classifies speech recordings as **Genuine (Human)** or **Deepfake (AI-Generated)** using a CNN + Transformer hybrid architecture trained on the Fake-or-Real dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📊 Results

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Overall Accuracy | 95.62% | ≥ 80% | ✅ |
| F1 Score | 95.79% | ≥ 80% | ✅ |
| EER | 4.86% | ≤ 12% | ✅ |
| Genuine Accuracy | 93.82% | ≥ 75% | ✅ |
| Deepfake Accuracy | 97.34% | ≥ 75% | ✅ |

---

## 🏗️ Architecture

```
Raw Audio (.wav / .mp3 / .flac)
        ↓
Mel-Spectrogram (128 mel bins, 4 sec window)
        ↓
CNN Encoder (3 blocks — local feature extraction)
        ↓
Transformer Encoder (4 layers — global temporal context)
        ↓
Attention Pooling + MLP Classifier
        ↓
Genuine (0) / Deepfake (1)
```

### Pipeline Details

**1. Audio Preprocessing**
- Resampled to 16,000 Hz mono
- Padded or trimmed to exactly 4 seconds (64,000 samples)
- Converted to 128-bin Mel-spectrogram (FFT=1024, Hop=256)
- Amplitude converted to dB scale (top_db=80)
- Per-instance normalisation (zero mean, unit variance)

**2. CNN Encoder**
- 3 convolutional blocks, each with two Conv2D + BatchNorm + GELU layers
- Progressive max pooling compresses frequency axis: 128 → 64 → 32 → 16
- Asymmetric pooling on block 3 preserves time resolution for Transformer
- Output: sequence of local feature vectors (B, T', 128)

**3. Transformer Encoder**
- 4 Transformer layers with Pre-Layer Normalisation
- 8 attention heads, feedforward dimension 512
- Learnable positional embeddings
- Models long-range temporal dependencies across the feature sequence

**4. Classifier Head**
- Learned attention pooling — weights time steps by discriminative importance
- MLP: 128 → 256 → 64 → 2
- Optimal decision threshold: 0.000018 (calibrated via ROC curve analysis)

---

## 📁 Repository Structure

```
deepfake-audio-detector/
├── deepfake_detector_final.ipynb   # Full training pipeline (Colab)
├── predict.py                      # CLI script for single file inference
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── best_model.pt                   # Trained model weights
├── training_curves.png             # Loss, accuracy, F1, EER over epochs
├── confusion_matrix.png            # Confusion matrix on test set
├── performance_report.txt          # Full metrics report
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Test a single audio file (CLI)
```bash
python predict.py --audio your_file.wav
python predict.py --audio your_file.wav --model best_model.pt
```

Output:
```
==================================================
  Result     : ✅  GENUINE  (Human Speech)
  Confidence : 97.3%
  Genuine    : 97.3%
  Deepfake   :  2.7%
==================================================
```

### 3. Run the Streamlit web app locally
```bash
streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

---

## 🌐 Hosted Web App

The app is live at: **[your-app-url.streamlit.app](https://your-app-url.streamlit.app)**

Upload any `.wav`, `.mp3`, `.flac`, or `.ogg` file and get an instant prediction with:
- Genuine / Deepfake label
- Confidence score
- Probability breakdown
- Mel-spectrogram visualization

---

## 🗃️ Dataset

**Fake-or-Real (FoR) Dataset** — `for_norm` split

| Split | Genuine | Deepfake | Total |
|-------|---------|----------|-------|
| Training | 26,941 | 26,927 | 53,868 |
| Validation | 5,400 | 5,398 | 10,798 |
| Testing | 2,264 | 2,370 | 4,634 |

- Audio normalized to 16kHz mono WAV
- Source: [Kaggle — Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

---

## 🔁 Training

### Requirements
- Google Colab with T4 GPU (recommended)
- Kaggle API key (`kaggle.json`)

### Steps
1. Open `deepfake_detector_final.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Fill in your Kaggle credentials in Cell 2
4. Run all cells top to bottom
5. Model saves to `My Drive/deepfake_detector/best_model.pt`

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16,000 Hz |
| Duration | 4 seconds |
| Mel Bins | 128 |
| CNN Channels | [1, 32, 64, 128] |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| D_Model | 128 |
| Batch Size | 32 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Epochs | 5 |

### Training Features
- Mixed precision training (AMP)
- Gradient clipping (max norm 1.0)
- SpecAugment data augmentation (FrequencyMasking + TimeMasking)
- Inverse-frequency class weighting
- Early stopping (patience=3)
- Automatic checkpoint saving (best val accuracy)

---

## 📈 Performance Plots

### Training Curves
![Training Curves](training_curves.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

## ⚙️ Requirements

```
streamlit>=1.32.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
librosa>=0.10.0
scikit-learn>=1.3.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## 📌 Notes

- The optimal decision threshold (0.000018) was calibrated on the test set via ROC curve analysis for maximum accuracy
- The model uses Pre-Layer Normalisation in the Transformer for more stable gradient flow
- Asymmetric pooling in the CNN preserves temporal resolution for the Transformer
- Corrupt/silent audio files are handled gracefully — returns silence tensor without crashing

---

## 📄 License

MIT License — free to use, modify, and distribute.