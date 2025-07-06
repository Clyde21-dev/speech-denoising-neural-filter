# Speech Denoising using Neural Adaptive Filter

A hands-on Python project that demonstrates how to remove biometrical (heartbeat & breathing) noise from speech using a CRNN-based (Convolutional Recurrent Neural Network) neural adaptive filter, with PyTorch and Librosa.

---

## Features

- Loads a sample clean speech file
- Generates synthetic biometrical noise (heartbeat & breathing)
- Mixes clean speech and noise at a specific SNR (e.g., 5 dB)
- Plots all waveforms and spectrograms
- Implements a CRNN (CNN + GRU + CNN) neural adaptive filter
- Denoises noisy speech in the spectrogram domain, then reconstructs the waveform
- Listen to clean, noisy, and denoised audio in the notebook

---

## Run in Google Colab

You can run this project directly in Google Colab (no local setup needed):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]

Just click the badge above, then select "Open in Playground" or "Run All" in Colab.

---

## Installation (for local run)

Install dependencies with:

```bash
pip install torchaudio librosa matplotlib ipython


Usage

Simply run the notebook or script.
All steps are included in a single file (.ipynb or .py):

Loads a clean speech audio sample

Generates biometrical noise (heartbeat & breathing)

Mixes speech + noise at target SNR

Visualizes signals

Builds and runs a CRNN-based denoiser

Displays & plays back denoised result

Code Flow
Imports & Setup:
All required libraries (PyTorch, torchaudio, librosa, matplotlib, etc.)

1. Load Clean Speech:
Uses torchaudio to download and load a short WAV file.

2. Generate Biometrical Noise:
Synthesizes heartbeat (low-freq sinusoids) and breathing (modulated noise).

3. Mix at SNR:
Scales noise and adds to speech at 5 dB SNR.

4. Plot & Playback:
Visualizes all waveforms and allows listening to each.

5. CRNN Model:
Defines a simple CRNN:
CNN encoder → GRU → CNN decoder (predicts a mask).

6. Spectrogram & Denoising:

Computes STFT of noisy speech

Normalizes for input to neural net

Neural net predicts a spectral mask

Applies mask to noisy magnitude spectrogram

Reconstructs waveform via ISTFT

7. Visualization:
Plots denoised waveform and spectrogram.

Example Outputs
Clean Speech:

Biometrical Noise:

Noisy Speech:

Denoised Output:


Requirements
Python 3.8+

torch

torchaudio

librosa

matplotlib

ipython (for audio playback in notebook)

Citation / Credit
Clean speech sample: Lab41-SRI-VOiCES

Inspired by recent deep learning work in neural speech enhancement.

License
MIT License

Author
Clydedev-21
klajdi.dhana1@gmail.com