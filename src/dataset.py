import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

from preprocessing import load_song, create_mashup, random_crop, audio_to_mel

SR = 22050


class GenreDataset(Dataset):
    def __init__(self, files, labels, mashup=False):
        self.files = files
        self.labels = labels
        self.mashup = mashup

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if self.mashup:
            audio, sr = create_mashup(self.files)
        else:
            audio, sr = load_song(self.files[idx])

        audio = random_crop(audio)
        mel = audio_to_mel(audio)

        mel = torch.tensor(mel).float().unsqueeze(0)
        label = torch.tensor(self.labels[idx])

        return mel, label


class MashupDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = librosa.load(self.files[idx], sr=SR)

        audio = random_crop(audio)
        mel = audio_to_mel(audio)

        mel = torch.tensor(mel).float().unsqueeze(0)

        return mel