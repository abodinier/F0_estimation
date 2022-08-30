import math
import torch
import librosa
import numpy as np
import torch.nn as nn

def F_hz2midi(freq, tuning_hz=440, eps=1e-9): 
    """Convert frequency in Hz to its MIDI representation

    Args:
        freq (float): Frequency in Hz
        tuning_hz (float, optional): Reference tuning frequency in Hz. Defaults to 440.
        eps (float, optional): eps for numerical stability. Defaults to 1e-9.

    Returns:
        float: MIDI note
    """
    
    return 12 * np.log2( (freq + eps) / tuning_hz) + 69


def F_midi2hz(midi, tuning_hz=440): 
    """Convert MIDI note to the corresponding frequency in Hz

    Args:
        midi (float): MIDI note
        tuning_hz (float, optional): Reference tuning. Defaults to 440.

    Returns:
        float: Frequency in Hz
    """
    
    return tuning_hz * (2**((midi - 69)/12))


def get_Fcs(n_harmonic, semitone_scale=2, low_midi=24, high_midi=94):
    """Get bands center frequency

    Args:
        n_harmonic (int): Number of harmonics
        semitone_scale (int, optional): Semitone scale. Defaults to 2.
        low_midi (int, optional): Lowest MIDI note of the spectrum. Defaults to 24 as in the paper (corresponding to C1 note).
        high_midi (int, optional): _description_. Defaults to 94 (MIDI note of the frequency: sample_rate / 2H).

    Returns:
        list: List of bands center frequencies. Number of bands = n_harmonic * (high_midi - low_midi) * semitone_scale
    """
    level = int((high_midi - low_midi) * semitone_scale)
    midi = np.linspace(low_midi, high_midi, level + 1)
    hz = F_midi2hz(midi[:-1])

    harmonic_hz = []
    for i in range(n_harmonic):
        harmonic_hz = np.concatenate((harmonic_hz, hz * (i+1)), dtype=np.float32)

    return harmonic_hz, level


class HarmonicFilter(nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=512,
        n_harmonic=6,
        semitone_scale=2,  # Quarter note
        ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_harmonic = n_harmonic
        self.semitone_scale = semitone_scale
        
        # BW estimation as in the paper: DATA-DRIVEN HARMONIC FILTERS FOR AUDIO REPRESENTATION LEARNING [Minz Won, 2020]
        self.bw_alpha = nn.Parameter(torch.tensor([0.1079]), requires_grad=True)
        self.bw_beta = nn.Parameter(torch.tensor([24.7]), requires_grad=True)
        self.bw_Q = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        # First we compute STFT of audio signal:
        self.spectrogram = lambda X: torch.tensor(
            [
                librosa.stft(
                    x, 
                    n_fft=n_fft,
                    window="hann"
                )
            for x in X],
            dtype=torch.float
        )
        self.amplitude_to_db = librosa.amplitude_to_db

        harmonic_hz, self.level = get_Fcs(
            n_harmonic, semitone_scale
        )

        self.f0 = torch.tensor(harmonic_hz)  # shape (n_band,)
        self.zero = torch.zeros(1)
        self.fft_bins = torch.linspace(0, self.sample_rate // 2, n_fft)

    def get_filterbank(self):
        # Compute bandwith for each filter
        bw = (self.bw_alpha * self.f0 + self.bw_beta) / self.bw_Q
        bw = bw.unsqueeze(0)  # (1, n_band)
        
        f0 = self.f0.unsqueeze(0)  # (1, n_band)
        
        fft_bins = self.fft_bins.unsqueeze(1)  # (n_bins, 1)

        # Triangular filter
        left_slope = torch.matmul(fft_bins, (2 / bw)) + 1 - (2 * f0 / bw)
        right_slope = torch.matmul(fft_bins, (-2 / bw)) + 1 + (2 * f0 / bw)
        
        fb = torch.max(self.zero, torch.min(right_slope, left_slope))
        return fb

    def forward(self, waveform):
        spectrogram = torch.tensor(self.spectrogram(waveform), dtype=torch.float)

        harmonic_fb = self.get_filterbank()
        
        # Compute "Mel Spectrogram" equivalent
        harmonic_spec = torch.matmul(spectrogram, harmonic_fb)

        batch, channels, length = harmonic_spec.size()
        harmonic_spec = harmonic_spec.view(batch, self.n_harmonic, self.level, length)
        # To dB
        harmonic_spec = self.amplitude_to_db(harmonic_spec)
        
        return harmonic_spec

class Frontend(nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=512,
        n_harmonic=6,
        semitone_scale=2,
        ):
        super().__init__()
        self.HFilter = HarmonicFilter(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
        )
        self.bn = nn.BatchNorm2d(n_harmonic)
    
    def forward(self, x):
        return self.bn(self.HFilter(x))


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding="same"):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Backend(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = Conv(input_channels, output_channels, (3, 3))
        self.conv2 = Conv(input_channels, output_channels, (3, 3))
        self.conv3 = Conv(input_channels, output_channels, (3, 3))
        self.conv4 = Conv(input_channels, output_channels, (3, 3))
        self.conv5 = Conv(input_channels, output_channels, (3, 3))
        self.conv6 = Conv(input_channels, output_channels, (3, 3))
        self.conv7 = Conv(input_channels, output_channels, (3, 3))
    
        self.dense = nn.Linear(input_channels, output_channels, bias=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        x = self.dense(x)
        x = nn.Sigmoid()(x)
        
        return x


class HarmonicCNN(nn.Module):
    def __init__(self,
                sample_rate=22050,
                n_channels=128,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50,
                n_harmonic=6,
                semitone_scale=2):
        super().__init__()

        # Front-End
        self.front_end = Frontend(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
        )
        self.backend = Backend()

    def forward(self, x):
        x = self.front_end(x)
        x = self.backend(x)

        return x

if __name__ == "__main__":
    sample_rate = 22050
    n_harmonic = 6
    hf = HarmonicFilter(n_harmonic=n_harmonic)
    x = np.random.rand(25000)[np.newaxis, :]
    out = hf.spectrogram(x)
    print(out.shape)