import os
import random
import numpy as np
import torchaudio
import librosa

class Augmentor:
    def __init__(self, musan_root, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_files = self._load_files(os.path.join(musan_root, "noise"))
        self.music_files = self._load_files(os.path.join(musan_root, "music"))
        self.speech_files = self._load_files(os.path.join(musan_root, "speech"))

    def _load_files(self, path):
        files = []
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".wav"):
                    files.append(os.path.join(path, f))
        return files

    def _load_random_wav(self, file_list, target_len):
        if not file_list:
            return np.zeros(target_len, dtype=np.float32)
        wav_path = random.choice(file_list)
        wav, sr = torchaudio.load(wav_path)
        wav = wav[0].numpy()
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        if len(wav) < target_len:
            shortage = target_len - len(wav)
            wav = np.pad(wav, (0, shortage), mode='wrap')
        else:
            start = random.randint(0, len(wav) - target_len)
            wav = wav[start:start + target_len]
        return wav.astype(np.float32)

    def add_noise(self, waveform, snr_low=0, snr_high=15, noise_type="noise"):
        target_len = len(waveform)
        if noise_type == "noise":
            noise = self._load_random_wav(self.noise_files, target_len)
        elif noise_type == "music":
            noise = self._load_random_wav(self.music_files, target_len)
        else:
            noise = self._load_random_wav(self.speech_files, target_len)

        # Chọn SNR ngẫu nhiên
        snr = random.uniform(snr_low, snr_high)
        amp_waveform = np.sqrt(np.mean(waveform**2))
        amp_noise = np.sqrt(np.mean(noise**2))
        noise = noise * (amp_waveform / (10**(snr / 20)) / (amp_noise + 1e-8))
        return waveform + noise

    def random_gain(self, waveform, min_gain=0.7, max_gain=1.3):
        gain = random.uniform(min_gain, max_gain)
        return waveform * gain

    def time_stretch(self, waveform, min_rate=0.9, max_rate=1.1):
        rate = random.uniform(min_rate, max_rate)
        return librosa.effects.time_stretch(waveform, rate)

    def pitch_shift(self, waveform, n_steps_low=-2, n_steps_high=2):
        n_steps = random.uniform(n_steps_low, n_steps_high)
        return librosa.effects.pitch_shift(waveform, sr=self.sample_rate, n_steps=n_steps)

    def apply(self, waveform):
        """
        waveform: numpy array [1, T] hoặc [T]
        """
        if waveform.ndim == 2:
            waveform = waveform[0]

        aug_choices = [
            lambda x: self.add_noise(x, noise_type="noise"),
            lambda x: self.add_noise(x, noise_type="music"),
            lambda x: self.add_noise(x, noise_type="speech"),
            self.random_gain,
            self.time_stretch,
            self.pitch_shift
        ]

        num_augs = random.randint(1, 3)
        chosen_augs = random.sample(aug_choices, num_augs)

        for aug in chosen_augs:
            waveform = aug(waveform)

        return np.clip(waveform, -1.0, 1.0).astype(np.float32)
