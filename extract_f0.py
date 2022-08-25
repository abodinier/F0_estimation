import sys
import warnings
import numpy as np

import torch
from torch import nn

from model import PaperModel

import librosa
import mirdata
import mir_eval
import soundfile

TARGET_SR = 22050
BINS_PER_SEMITONE = 3
N_OCTAVES = 6
FMIN = 32.7
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
HOP_LENGTH = 512  # 23 ms hop

CQT_FREQUENCIES = librosa.cqt_frequencies(N_BINS, FMIN, BINS_PER_OCTAVE)

mirdata.initialize("medleydb_pitch")

def load_audio(audio_path):
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    return y

def compute_hcqt(audio):
    cqt = librosa.cqt(
        audio,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    cqt = (1.0 / 80.0) * librosa.amplitude_to_db(np.abs(cqt), ref=1.0) + 1.0

    hcqt = librosa.interp_harmonics(cqt, CQT_FREQUENCIES, [0.5, 1, 2, 3, 4])  # there are 5 harmonics ranges this is the dimension 0
    return hcqt.transpose([0, 2, 1])

def get_cqt_times(n_bins):
    return librosa.frames_to_time(np.arange(n_bins), sr=TARGET_SR, hop_length=HOP_LENGTH)

def sonify_outputs(save_path, times, freqs, voicing):
    y = mir_eval.sonify.pitch_contour(times, freqs, 8000, amplitudes=voicing)
    soundfile.write(save_path, y, 8000)

def estimate_f0(model, audio):
    hcqt = compute_hcqt(audio)
    n_times = hcqt.shape[1]

    slice_size = 200
    outputs = []

    with torch.no_grad():
        for i in np.arange(0, n_times, step=slice_size):
            hcqt_tensor = torch.tensor(hcqt[np.newaxis, :, i : i + slice_size, :]).float()
            predicted_salience = model(hcqt_tensor)
            predicted_salience = nn.Sigmoid()(predicted_salience).detach()
            outputs.append(predicted_salience)

    unwrapped_prediction = np.hstack(outputs)[0, :n_times, :, 0].astype(float)

    transition_matrix = librosa.sequence.transition_local(len(CQT_FREQUENCIES), 5)
    predicted_pitch_idx = librosa.sequence.viterbi(unwrapped_prediction.T, transition_matrix)

    predicted_pitch = np.array([CQT_FREQUENCIES[f] for f in predicted_pitch_idx])
    predicted_salience = np.array(
        [unwrapped_prediction[i, f] for i, f in enumerate(predicted_pitch_idx)]
    )

    times = get_cqt_times(n_times)

    est_times, est_freqs, est_voicing = mirdata.annotations.F0Data(
        times, "s", predicted_pitch, "hz", predicted_salience, "likelihood"
    ).to_mir_eval()

    return est_times, est_freqs, est_voicing

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    
    MODEL_PATH = sys.argv[1]
    TRACK_PATH = sys.argv[2]
    try:
        DESTINATION = sys.argv[3]
    except:
        DESTINATION = f"./estimated_{TRACK_PATH}"
    
    
    model = PaperModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"Model successfully loaded from {MODEL_PATH}")

    audio = load_audio(TRACK_PATH)
    print(f"Audio loaded ({TRACK_PATH})")

    print("Running salience pitch estimation")
    est_times, est_freqs, est_voicing = estimate_f0(model, audio)
    print("Done")
    
    print("Sonifying output...")
    sonify_outputs(
        DESTINATION,
        est_times,
        est_freqs,
        est_voicing,
    )
    print("Success!")