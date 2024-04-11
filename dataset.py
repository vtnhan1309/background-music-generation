import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from stft import TacotronSTFT


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output



class Text2AudioDataset(Dataset):
    def __init__(self, file, shuffle=True):
        self.duration = 10.24
        self.trim_wav = False
        self.pad_wav_start_sample = 0
        
        # Read from the json config
        self.melbins = 64
        self.sampling_rate = 16000
        self.hopsize = 160
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.mixup = 0.0

        self.build_dsp()

        self.items = []
        with open(file, 'r') as file_obj:
            lines = file_obj.readlines()

        for line in lines:
            t = json.loads(line)
            self.items.append({
                'path': t['location'],
                'text': t['captions']
            })

        random.seed(1234)
        if shuffle:
            random.shuffle(self.items)


    def build_dsp(self):
        self.mel_basis = {}
        self.hann_window = {}

        self.filter_length = 1024
        self.hop_length = 160
        self.win_length = 1024
        self.n_mel = 64
        self.sampling_rate = 16000
        self.mel_fmin = 0
        self.mel_fmax = 8000

        self.STFT = TacotronSTFT(
            self.filter_length,
            self.hop_length,
            self.win_length,
            self.n_mel,
            self.sampling_rate,
            self.mel_fmin,
            self.mel_fmax,
        )

    def __len__(self):
        return len(self.items)

    def get_num_instances(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        log_mel_spec, stft, mix_lambda, waveform, random_start = self.read_audio_file(item['path'])
        log_mel_spec = log_mel_spec.unsqueeze(0)
        return item['text'], log_mel_spec,


    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val
    

    def random_segment_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        for i in range(10):
            random_start = int(self.random_uniform(0, waveform_length - target_length))
            if torch.max(
                torch.abs(waveform[:, random_start : random_start + target_length])
                > 1e-4
            ):
                break

        return waveform[:, random_start : random_start + target_length], random_start
    

    def resample(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        return waveform
    

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5
    

    def read_wav_file(self, filename):
        # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
        waveform, sr = torchaudio.load(filename)

        #TODO: comment random segment and retrain
        waveform, random_start = self.random_segment_wav(
            waveform, target_length=int(sr * self.duration)
        )

        waveform = self.resample(waveform, sr)
        # random_start = int(random_start * (self.sampling_rate / sr))

        waveform = waveform.numpy()[0, ...]

        waveform = self.normalize_wav(waveform)

        waveform = waveform[None, ...]
        waveform = self.pad_wav(
            waveform, target_length=int(self.sampling_rate * self.duration)
        )
        return waveform, 0
    

    def pad_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if self.pad_wav_start_sample is None:
            rand_start = int(self.random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav
    
    
    def read_audio_file(self, filename, filename2=None):
        waveform, random_start = self.read_wav_file(filename)
        mix_lambda = 0.0
        log_mel_spec, stft = self.wav_feature_extraction(waveform)
        return log_mel_spec, stft, mix_lambda, waveform, random_start
    

    def wav_feature_extraction(self, waveform):
        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        # log_mel_spec, stft, energy = Audio.tools.get_mel_from_wav(waveform, self.STFT)[0]
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))

        log_mel_spec = torch.FloatTensor(log_mel_spec.T)
        stft = torch.FloatTensor(stft.T)

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)
        return log_mel_spec, stft
    

    def mel_spectrogram_train(self, y):
        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                self.sampling_rate,
                self.filter_length,
                self.n_mel,
                self.mel_fmin,
                self.mel_fmax,
            )
            self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(
                y.device
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.filter_length - self.hop_length) / 2),
                int((self.filter_length - self.hop_length) / 2),
            ),
            mode="reflect",
        )

        y = y.squeeze(1)

        stft_spec = torch.stft(
            y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[str(y.device)],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)

        mel = spectral_normalize_torch(
            torch.matmul(
                self.mel_basis[str(self.mel_fmax) + "_" + str(y.device)], stft_spec
            )
        )

        return mel[0], stft_spec[0]

    
    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec
