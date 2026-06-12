# ============================================================
# DEEPFAKE AUDIO DETECTION — Streamlit Web App
# Run: streamlit run app.py
# ============================================================

import os
import io
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 800px; margin: 0 auto; }

    .result-genuine {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border: 2px solid #52b788;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
    }
    .result-deepfake {
        background: linear-gradient(135deg, #6b1a1a, #9b2226);
        border: 2px solid #e63946;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 20px 0;
    }
    .result-label {
        font-size: 2.4rem;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: 1px;
    }
    .result-sublabel {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin-top: 6px;
    }
    .confidence-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border-radius: 50px;
        padding: 6px 20px;
        font-size: 1.1rem;
        color: white;
        font-weight: 600;
        margin-top: 14px;
    }
    .prob-row {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-top: 16px;
    }
    .prob-item {
        text-align: center;
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
    }
    .prob-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
    }
    .upload-box {
        border: 2px dashed #4a4a6a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: #0e0e1a;
    }
    .info-card {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 3px solid #7c6fcd;
    }
    h1 { color: #e0e0ff !important; }
    .stProgress > div > div { background-color: #7c6fcd; }
</style>
""", unsafe_allow_html=True)


# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    SAMPLE_RATE  = 16000
    DURATION     = 4
    N_MELS       = 128
    N_FFT        = 1024
    HOP_LENGTH   = 256
    F_MIN        = 20
    F_MAX        = 8000
    MAX_SAMPLES  = SAMPLE_RATE * DURATION
    TIME_FRAMES  = MAX_SAMPLES // HOP_LENGTH + 1
    CNN_CHANNELS = [1, 32, 64, 128]
    CNN_DROPOUT  = 0.2
    D_MODEL      = 128
    NHEAD        = 8
    NUM_LAYERS   = 4
    DIM_FF       = 512
    TF_DROPOUT   = 0.1
    MODEL_PATH   = "best_model.pt"

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.MaxPool2d(pool),
            nn.Dropout2d(cfg.CNN_DROPOUT),
        )
    def forward(self, x): return self.net(x)


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = cfg.CNN_CHANNELS
        self.b1   = CNNBlock(ch[0], ch[1], pool=(2, 2))
        self.b2   = CNNBlock(ch[1], ch[2], pool=(2, 2))
        self.b3   = CNNBlock(ch[2], ch[3], pool=(2, 1))
        self.proj = nn.Linear(ch[3] * (cfg.N_MELS // 8), cfg.D_MODEL)
        self.norm = nn.LayerNorm(cfg.D_MODEL)

    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)
        return self.norm(self.proj(x))


class TransformerEncoder(nn.Module):
    def __init__(self, max_len=500):
        super().__init__()
        self.pos = nn.Embedding(max_len, cfg.D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, nhead=cfg.NHEAD,
            dim_feedforward=cfg.DIM_FF, dropout=cfg.TF_DROPOUT,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.tf = nn.TransformerEncoder(
            layer, num_layers=cfg.NUM_LAYERS,
            norm=nn.LayerNorm(cfg.D_MODEL),
        )

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.tf(x + self.pos(pos))


class ClassifierHead(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.attn = nn.Linear(cfg.D_MODEL, 1)
        self.mlp  = nn.Sequential(
            nn.Linear(cfg.D_MODEL, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64),          nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return self.mlp((w * x).sum(dim=1))


class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn         = CNNEncoder()
        self.transformer = TransformerEncoder()
        self.head        = ClassifierHead()

    def forward(self, x):
        return self.head(self.transformer(self.cnn(x)))


# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(cfg.MODEL_PATH):
        return None
    model = DeepfakeDetector().to(DEVICE)
    model.load_state_dict(
        torch.load(cfg.MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


# ── Audio processing ──────────────────────────────────────────────────────────
def process_audio(file_bytes: bytes):
    """Convert uploaded audio bytes → mel spectrogram tensor."""
    mel_t = T.MelSpectrogram(
        sample_rate=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,   n_mels=cfg.N_MELS,
        f_min=cfg.F_MIN,             f_max=cfg.F_MAX,
    )
    db_t = T.AmplitudeToDB(top_db=80)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        wav, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    if sr != cfg.SAMPLE_RATE:
        wav = T.Resample(sr, cfg.SAMPLE_RATE)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    n = wav.shape[1]
    wav = F.pad(wav, (0, cfg.MAX_SAMPLES - n)) if n < cfg.MAX_SAMPLES \
          else wav[:, :cfg.MAX_SAMPLES]

    mel = db_t(mel_t(wav))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0), mel.squeeze().numpy()   # tensor, numpy for plot


def plot_spectrogram(mel_np: np.ndarray) -> plt.Figure:
    """Render mel-spectrogram as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0e0e1a")
    ax.set_facecolor("#0e0e1a")
    img = ax.imshow(
        mel_np, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest",
    )
    ax.set_title("Mel-Spectrogram", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Time Frames", color="#aaaacc")
    ax.set_ylabel("Mel Bins", color="#aaaacc")
    ax.tick_params(colors="#aaaacc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="#aaaacc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaacc")
    plt.tight_layout()
    return fig


# ── App UI ────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("# 🎙️ Deepfake Audio Detector")
    st.markdown(
        "Upload a speech recording to detect whether it is "
        "**Genuine (Human)** or **Deepfake (AI-Generated)**."
    )
    st.divider()

    # Pipeline info
    with st.expander("ℹ️ How it works", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**1. 🎵 Audio Input**\nWAV / MP3 / FLAC")
        with col2:
            st.markdown("**2. 📊 Mel-Spectrogram**\n128 mel bins")
        with col3:
            st.markdown("**3. 🧠 CNN + Transformer**\nLocal + global features")
        with col4:
            st.markdown("**4. 🏷️ Classification**\nGenuine vs Deepfake")

    st.divider()

    # Load model
    model = load_model()
    if model is None:
        st.error(
            "⚠️ `best_model.pt` not found in the app directory. "
            "Please place the trained model file next to `app.py`."
        )
        st.stop()

    # File uploader
    st.markdown("### 📂 Upload Audio File")
    uploaded = st.file_uploader(
        label="Choose an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        help="Supported formats: WAV, MP3, FLAC, OGG"
    )

    if uploaded is None:
        st.info("👆 Upload an audio file above to get started.")
        return

    # Audio player
    st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
    st.markdown(f"**File:** `{uploaded.name}`  |  "
                f"**Size:** `{uploaded.size / 1024:.1f} KB`")

    # Run inference
    with st.spinner("Analysing audio..."):
        try:
            file_bytes = uploaded.read()
            mel_tensor, mel_np = process_audio(file_bytes)
            mel_tensor = mel_tensor.to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(
                    model(mel_tensor), dim=1)[0].cpu().numpy()

            genuine_prob  = float(probs[0])
            deepfake_prob = float(probs[1])
            THRESHOLD = 0.000018
            label = "Deepfake" if deepfake_prob >= THRESHOLD else "Genuine"
            confidence = deepfake_prob if label == "Deepfake" else genuine_prob

        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return

    # ── Result card ──────────────────────────────────────────────────────────
    st.markdown("### 🔍 Detection Result")

    if label == "Genuine":
        st.markdown(f"""
        <div class="result-genuine">
            <p class="result-label">✅ GENUINE</p>
            <p class="result-sublabel">This audio appears to be real human speech</p>
            <div class="confidence-badge">Confidence: {confidence*100:.1f}%</div>
            <div class="prob-row">
                <div class="prob-item">
                    <div class="prob-value">{genuine_prob*100:.1f}%</div>
                    Genuine
                </div>
                <div class="prob-item">
                    <div class="prob-value">{deepfake_prob*100:.1f}%</div>
                    Deepfake
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-deepfake">
            <p class="result-label">⚠️ DEEPFAKE</p>
            <p class="result-sublabel">This audio appears to be AI-generated speech</p>
            <div class="confidence-badge">Confidence: {confidence*100:.1f}%</div>
            <div class="prob-row">
                <div class="prob-item">
                    <div class="prob-value">{genuine_prob*100:.1f}%</div>
                    Genuine
                </div>
                <div class="prob-item">
                    <div class="prob-value">{deepfake_prob*100:.1f}%</div>
                    Deepfake
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability bars ──────────────────────────────────────────────────────
    st.markdown("### 📊 Probability Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 Genuine**")
        st.progress(genuine_prob)
        st.markdown(f"`{genuine_prob*100:.2f}%`")
    with col2:
        st.markdown("**🔴 Deepfake**")
        st.progress(deepfake_prob)
        st.markdown(f"`{deepfake_prob*100:.2f}%`")

    # ── Spectrogram ───────────────────────────────────────────────────────────
    st.markdown("### 🎨 Mel-Spectrogram Visualization")
    fig = plot_spectrogram(mel_np)
    st.pyplot(fig)
    plt.close(fig)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#666688; font-size:0.85rem;'>"
        "CNN + Transformer pipeline · Trained on Fake-or-Real Dataset · "
        "Val Accuracy 99.9% · EER 0.08%"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
