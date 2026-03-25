import os
import numpy as np
import librosa
import random

SR = 22050

# -------------------------------
# Load stems and combine
# -------------------------------
def load_song(song_folder):
    stems = ["drums.wav", "bass.wav", "vocals.wav", "other.wav"]

    audio_sum = None

    for stem in stems:
        path = os.path.join(song_folder, stem)
        if not os.path.exists(path):
            continue

        y, sr = librosa.load(path, sr=SR)

        if audio_sum is None:
            audio_sum = y
        else:
            min_len = min(len(audio_sum), len(y))
            audio_sum = audio_sum[:min_len] + y[:min_len] * 0.25

    return audio_sum, SR


# -------------------------------
# Create cross-song mashup
# -------------------------------
def create_mashup(song_list):
    stems = ["drums.wav", "bass.wav", "vocals.wav", "other.wav"]

    audio_sum = None

    for stem in stems:
        song = random.choice(song_list)
        path = os.path.join(song, stem)

        if not os.path.exists(path):
            continue

        y, sr = librosa.load(path, sr=SR)

        if audio_sum is None:
            audio_sum = y
        else:
            min_len = min(len(audio_sum), len(y))
            audio_sum = audio_sum[:min_len] + y[:min_len] * 0.25

    return audio_sum, SR


# -------------------------------
# Add noise
# -------------------------------
def add_noise(audio, noise_files, noise_level=0.02):
    noise_file = random.choice(noise_files)
    noise_audio, _ = librosa.load(noise_file, sr=SR)

    if len(noise_audio) < len(audio):
        noise_audio = np.pad(noise_audio, (0, len(audio) - len(noise_audio)))
    else:
        noise_audio = noise_audio[:len(audio)]

    audio = audio + noise_level * noise_audio
    return audio


# -------------------------------
# Random crop
# -------------------------------
def random_crop(audio, seconds=5):
    crop_size = SR * seconds

    if len(audio) <= crop_size:
        return audio[:crop_size]

    start = np.random.randint(0, len(audio) - crop_size)
    return audio[start:start + crop_size]


# -------------------------------
# Mel Spectrogram
# -------------------------------
def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=64
    )

    mel = librosa.power_to_db(mel)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)

    if mel.shape[1] > 512:
        mel = mel[:, :512]
    else:
        pad = 512 - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad)))

    return mel


# -------------------------------
# MFCC features (Milestone-2)
# -------------------------------
def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20)
    return np.mean(mfcc, axis=1)