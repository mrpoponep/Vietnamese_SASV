import os
import torch
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SASVDataset(Dataset):
    def __init__(self, data_root, sample_rate=16000, transform=None, augmentor=None, num_frames=200):
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.transform = transform or torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, hop_length=160, n_mels=80
        )
        self.augmentor = augmentor
        self.num_frames = num_frames
        self.segment_length = self.num_frames * 160 + 240  # tương đương ~2s

        self.data = self._prepare_data()

    def _prepare_data(self):
        data = []
        speakers = os.listdir(self.data_root)

        for speaker in speakers:
            speaker_path = os.path.join(self.data_root, speaker)
            bonafide_path = os.path.join(speaker_path, 'bonafide')
            spoof_path = os.path.join(speaker_path, 'spoof')

            bonafide_files = []
            spoof_files = []

            # Kiểm tra tồn tại thư mục trước khi đọc
            if os.path.isdir(bonafide_path):
                bonafide_files = [
                    os.path.join(bonafide_path, f)
                    for f in os.listdir(bonafide_path)
                    if f.endswith(".wav")
                ]

            if os.path.isdir(spoof_path):
                spoof_files = [
                    os.path.join(spoof_path, f)
                    for f in os.listdir(spoof_path)
                    if f.endswith(".wav")
                ]

            # Nếu không có bonafide thì bỏ qua vì cần để làm enrollment
            if len(bonafide_files) == 0:
                continue

            # Gán nhãn: bonafide = 1, spoof = 0
            for test_path in spoof_files + bonafide_files:
                label = 1 if test_path in bonafide_files else 0
                enr_path = random.choice(bonafide_files)
                data.append((enr_path, test_path, label))

        return data

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resample(waveform)
        return waveform[0]  # [T]

    def _crop_or_pad(self, waveform):
        length = waveform.shape[0]
        target_len = self.segment_length

        if length < target_len:
            shortage = target_len - length
            waveform = torch.cat([waveform, waveform[:shortage]], dim=0)
        else:
            start = random.randint(0, length - target_len)
            waveform = waveform[start:start + target_len]
        return waveform

    def __getitem__(self, idx):
        enr_path, tst_path, label = self.data[idx]
        enr_waveform = self._load_audio(enr_path)
        tst_waveform = self._load_audio(tst_path)

        # Cắt đoạn (giống ECAPA)
        enr_waveform = self._crop_or_pad(enr_waveform)
        tst_waveform = self._crop_or_pad(tst_waveform)

        # Augment test waveform
        if self.augmentor:
            tst_np = tst_waveform.numpy()
            tst_np = np.expand_dims(tst_np, axis=0)  # [1, T]
            tst_np = self.augmentor.apply(tst_np)
            tst_waveform = torch.from_numpy(tst_np[0])

        # Mel features cho ECAPA
        enr_feat = self.transform(enr_waveform).transpose(0, 1)  # [T, F]
        tst_feat = self.transform(tst_waveform).transpose(0, 1)  # [T, F]

        return {
            'tst_waveform': tst_waveform,   # [T]
            'enr_feature': enr_feat,        # [T, F]
            'tst_feature': tst_feat,        # [T, F]
            'label': label
        }

    def __len__(self):
        return len(self.data)
