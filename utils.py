import torch
import librosa
import mirdata
import mir_eval
import soundfile
import numpy as np
import matplotlib.pyplot as plt

TARGET_SR = 22050
BINS_PER_SEMITONE = 3
N_OCTAVES = 6
FMIN = 32.7
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
HOP_LENGTH = 512  # 23 ms hop
N_TIME_FRAMES = 50  # 1.16 seconds
H_RANGE = [0.5, 1, 2, 3, 4]

CQT_FREQUENCIES = librosa.cqt_frequencies(N_BINS, FMIN, BINS_PER_OCTAVE)
TRANSITION_MATRIX = librosa.sequence.transition_local(N_BINS, len(H_RANGE))


def get_cqt_times(n_bins):
    return librosa.frames_to_time(np.arange(n_bins), sr=TARGET_SR, hop_length=HOP_LENGTH)


TIMES_S = get_cqt_times(N_TIME_FRAMES)

mirdata.initialize("medleydb_pitch")

def visualize(model, hcqt, salience, predicted_salience=None, n=3):
    if predicted_salience is None:
        predicted_salience = model(hcqt).detach()
    
    fig = plt.figure(figsize=(15, 10))
    
    for i in range(n):

        # plot the input
        plt.subplot(n, 3, 1 + (n * i))
        # use channel 1 of the hcqt, which corresponds to h=1
        plt.imshow(hcqt[i, 1, :, :].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title(f"HCQT")
        plt.axis("tight")

        # plot the target salience
        plt.subplot(n, 3, 2 + (n * i))
        plt.imshow(salience[i, :, :, 0].T, origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Target")
        plt.axis("tight")

        # plot the predicted salience
        plt.subplot(n, 3, 3 + (n * i))
        plt.imshow(torch.sigmoid(predicted_salience[i, :, :, 0].T), origin="lower", cmap="magma")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("Prediction")
        plt.axis("tight")

    plt.tight_layout()
    return fig


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

    hcqt = librosa.interp_harmonics(cqt, CQT_FREQUENCIES, H_RANGE)  # there are 5 harmonics ranges this is the dimension 0
    return hcqt.transpose([0, 2, 1])


def sonify_outputs(save_path, times, freqs, voicing):
    y = mir_eval.sonify.pitch_contour(times, freqs, 8000, amplitudes=voicing)
    soundfile.write(save_path, y, 8000)


def extract_freqs(transition_matrix, times, salience_2D):
    pitch = librosa.sequence.viterbi(salience_2D.T, transition_matrix)
    
    pitch_hz = np.array([CQT_FREQUENCIES[f] for f in pitch])
    salience_1D = np.array(
        [salience_2D[i, f] for i, f in enumerate(pitch)]
    )
    
    estimated_f0 = mirdata.annotations.F0Data(
        times, "s", pitch_hz, "hz", salience_1D, "likelihood"
    )
    return estimated_f0.to_mir_eval()


def evaluate(model, data):
    model.eval()

    results = {}

    for batch in data:
        hcqt, target_saliences = batch
        target_saliences = torch.transpose(target_saliences, 0, 2)
        target_saliences = target_saliences[:, :, :, 0].T.detach().numpy().astype(float)
        
        predicted_saliences = model.predict(hcqt).detach().numpy().astype(float)[:, :, :, 0]
        
        for predicted_salience, target_salience in zip(
            predicted_saliences, target_saliences
        ):
            est_times, est_freqs, est_voicing = extract_freqs(
                TRANSITION_MATRIX,
                TIMES_S,
                predicted_salience
            )
            target_times, target_freqs, target_voicing = extract_freqs(
                TRANSITION_MATRIX,
                TIMES_S,
                target_salience
            )
            
            res = mir_eval.melody.evaluate(
                    target_times, target_freqs, est_times, est_freqs, est_voicing=est_voicing
                )
            
            for k, v in res.items():
                if k in results:
                    results[k].append(v)
                else:
                    results[k] = [v,]
    
    return results