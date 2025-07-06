# Install dependencies
# !pip install torchaudio librosa matplotlib ipython

# %% Imports
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# reproducibility
torch.manual_seed(42)

# %% 1. Load a sample clean speech file
asset = torchaudio.utils.download_asset(
    "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav"
)
waveform, sample_rate = torchaudio.load(asset)
waveform = waveform.mean(dim=0)  # to mono
print(f"Clean speech: {waveform.shape[0]} samples @ {sample_rate} Hz")

# %% 2. Generate synthetic biometrical noise
def generate_biometrical_noise(length, sr):
    t = torch.linspace(0, length/sr, length)
    # heartbeat: 1.5 Hz + harmonic
    heartbeat = 0.3 * torch.sin(2 * np.pi * 1.5 * t)
    heartbeat += 0.1 * torch.sin(2 * np.pi * 3.0 * t)
    # breathing: low-freq modulated white noise
    breathing = 0.2 * torch.randn(length) * torch.sigmoid(torch.sin(2 * np.pi * 0.3 * t))
    return heartbeat + breathing

noise = generate_biometrical_noise(waveform.shape[0], sample_rate)
print(f"Biometrical noise: {noise.shape[0]} samples")

# %% 3. Mix clean + noise at a target SNR (5 dB)
def mix_at_snr(clean, noise, snr_db):
    p_clean = clean.pow(2).mean()
    p_noise = noise.pow(2).mean()
    snr_lin = 10 ** (snr_db / 10)
    scale = torch.sqrt(p_clean / (p_noise * snr_lin))
    return clean + scale * noise

noisy_speech = mix_at_snr(waveform, noise, snr_db=5)

# %% 4. Plot waveforms
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(waveform.numpy());    plt.title("Clean Speech")
plt.subplot(3, 1, 2)
plt.plot(noise.numpy());       plt.title("Biometrical Noise")
plt.subplot(3, 1, 3)
plt.plot(noisy_speech.numpy());plt.title("Noisy Speech (5 dB SNR)")
plt.tight_layout(); plt.show()

print("▶ Clean speech playback:")
display(Audio(waveform.numpy(), rate=sample_rate))
print("▶ Noisy speech playback:")
display(Audio(noisy_speech.numpy(), rate=sample_rate))

# %% 5. Define the CRNN-based neural adaptive filter
class NeuralAdaptiveFilter(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN encoder: freq×time → channels×hf×ht
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.ReLU()
        )
        # GRU on flattened freq×chan
        self.gru = nn.GRU(input_size=32* (n_fft//4 + 1), hidden_size=64, bidirectional=True, batch_first=True)
        # CNN decoder: back to single-channel mask
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 16, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, freq_bins, time_frames)
        x = x.unsqueeze(1)  # → (batch, 1, freq, time)
        enc = self.encoder(x)
        b, c, f, t = enc.shape
        # prepare for RNN: (batch, time, features)
        r = enc.permute(0, 3, 1, 2).reshape(b, t, c*f)
        out, _ = self.gru(r)              # (batch, time, 2*h)
        # map back to decoder input shape
        dec_in = out.reshape(b, 128, 1, t)
        mask = self.decoder(dec_in)       # → (batch, 1, freq, time)
        return mask.squeeze(1)            # → (batch, freq, time)

# Instantiate
model = NeuralAdaptiveFilter()
print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")

# %% 6. Compute spectrograms, denoise, and reconstruct
n_fft     = 512
hop_length= 128

# a) STFT of noisy signal (using librosa)
y_noisy = noisy_speech.numpy()
D_noisy = librosa.stft(y_noisy, n_fft=n_fft, hop_length=hop_length)  # (freq_bins, time)

# b) Convert amplitude → dB
S_db = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)

# c) Plot spectrogram (explicit n_fft avoids divide-by-zero)
plt.figure(figsize=(12,4))
librosa.display.specshow(
    S_db,
    sr=sample_rate,
    hop_length=hop_length,
    n_fft=n_fft,
    x_axis='time',
    y_axis='log'
)
plt.colorbar(format='%+2.0f dB')
plt.title("Noisy Speech Spectrogram")
plt.tight_layout(); plt.show()

# d) Normalize for model input
S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-9)
mag_tensor = torch.from_numpy(S_norm).unsqueeze(0).float()  # (1, freq, time)

# e) Predict mask & apply
with torch.no_grad():
    mask = model(mag_tensor).squeeze(0).numpy()           # (freq, time)
denoised_mag = mask * np.abs(D_noisy)

# f) ISTFT back to time domain
phase = np.angle(D_noisy)
D_denoised = denoised_mag * np.exp(1j * phase)
denoised_waveform = librosa.istft(D_denoised, hop_length=hop_length)

# %% 7. Plot & listen to denoised result
plt.figure(figsize=(12, 3))
plt.plot(denoised_waveform)
plt.title("Denoised Speech Waveform")
plt.tight_layout(); plt.show()

print("▶ Noisy input playback:")
display(Audio(y_noisy, rate=sample_rate))
print("▶ Denoised output playback:")
display(Audio(denoised_waveform, rate=sample_rate))

# %% 8. Conclusion
# - We fixed the spectrogram plotting by passing `n_fft` explicitly.
# - We used `librosa.amplitude_to_db` to avoid manual log/normalization pitfalls.
# - The CRNN mask-based approach then cleans the speech of biometrical noise.