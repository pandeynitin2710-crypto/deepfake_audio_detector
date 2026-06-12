# ============================================================
# DEEPFAKE AUDIO DETECTION — CLI Prediction Script
# Usage:
#   python predict.py --audio your_file.wav
#   python predict.py --audio your_file.wav --model best_model.pt
# ============================================================

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


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

cfg = Config()


# ── Model Architecture ────────────────────────────────────────────────────────
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


# ── Audio preprocessing ───────────────────────────────────────────────────────
def load_audio(path: str) -> torch.Tensor:
    """Load audio file and convert to normalised mel-spectrogram."""
    mel_t = T.MelSpectrogram(
        sample_rate=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,   n_mels=cfg.N_MELS,
        f_min=cfg.F_MIN,             f_max=cfg.F_MAX,
    )
    db_t = T.AmplitudeToDB(top_db=80)

    wav, sr = torchaudio.load(path)
    if sr != cfg.SAMPLE_RATE:
        wav = T.Resample(sr, cfg.SAMPLE_RATE)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    n = wav.shape[1]
    wav = F.pad(wav, (0, cfg.MAX_SAMPLES - n)) if n < cfg.MAX_SAMPLES \
          else wav[:, :cfg.MAX_SAMPLES]

    mel = db_t(mel_t(wav))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.unsqueeze(0)   # (1, 1, N_MELS, T)


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(audio_path: str, model_path: str, device: torch.device) -> dict:
    """
    Run inference on a single audio file.

    Returns:
        dict with keys: label, confidence, genuine_prob, deepfake_prob
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = DeepfakeDetector().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Preprocess + infer
    mel = load_audio(audio_path).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(mel), dim=1)[0].cpu().numpy()

    THRESHOLD = 0.000018
    label = "Deepfake" if probs[1] >= THRESHOLD else "Genuine"
    return {
        "label"        : label,
        "confidence"   : float(probs.max()),
        "genuine_prob" : float(probs[0]),
        "deepfake_prob": float(probs[1]),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Audio Detector — classify a speech file as "
                    "Genuine (Human) or Deepfake (AI-Generated)."
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to audio file (.wav / .flac / .mp3 / .ogg)"
    )
    parser.add_argument(
        "--model", default="best_model.pt",
        help="Path to trained model weights (default: best_model.pt)"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on (default: auto)"
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nDevice : {device}")
    print(f"Model  : {args.model}")
    print(f"Audio  : {args.audio}")

    result = predict(args.audio, args.model, device)

    sep = "=" * 50
    label_display = (
        f"✅  GENUINE  (Human Speech)"
        if result["label"] == "Genuine"
        else f"⚠️   DEEPFAKE (AI-Generated)"
    )
    print(f"\n{sep}")
    print(f"  Result     : {label_display}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"  Genuine    : {result['genuine_prob']*100:.1f}%")
    print(f"  Deepfake   : {result['deepfake_prob']*100:.1f}%")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
