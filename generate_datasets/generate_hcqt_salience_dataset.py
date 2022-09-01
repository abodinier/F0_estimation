import os
import mirdata
import librosa
import argparse
from re import L
import numpy as np


TARGET_SR = 22050
BINS_PER_SEMITONE = 5
N_OCTAVES = 6
FMIN = 32.7
BINS_PER_OCTAVE = 12 * BINS_PER_SEMITONE
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
HOP_LENGTH = 512  # 23 ms hop
N_TIME_FRAMES = 50  # 1.16 seconds
N_AUDIO_SAMPLES = HOP_LENGTH * N_TIME_FRAMES
N_EXAMPLES_PER_TRACK = 100

CQT_FREQUENCIES = librosa.cqt_frequencies(N_BINS, FMIN, BINS_PER_OCTAVE)


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

    hcqt = librosa.interp_harmonics(cqt, CQT_FREQUENCIES, [0.5, 1, 2, 3, 4])
    return hcqt.transpose([0, 2, 1])


def get_cqt_times(n_bins):
    return librosa.frames_to_time(np.arange(n_bins), sr=TARGET_SR, hop_length=HOP_LENGTH)


def generate_examples(split):
    medleydb_pitch = mirdata.initialize("medleydb_pitch")
    
    split_track_ids = medleydb_pitch.get_random_track_splits([0.9, 0.1])
    
    if split == "train":
        track_ids = split_track_ids[0]
    else:
        track_ids = split_track_ids[1]

    total_track_ids = len(track_ids)
    for i, track_id in enumerate(track_ids):
        print(f"Track {i+1}/{total_track_ids}: {track_id}")
        
        track = medleydb_pitch.track(track_id)
        
        y = load_audio(track.audio_path)
        
        yield (y, track.pitch)


def generate_samples_from_example(audio, pitch_data):
    hcqt = compute_hcqt(audio)
    n_times = hcqt.shape[1]
    times = get_cqt_times(n_times)

    target_salience = pitch_data.to_matrix(times, "s", CQT_FREQUENCIES, "hz")

    all_time_indexes = np.arange(0, n_times - N_TIME_FRAMES)
    time_indexes = np.random.choice(all_time_indexes, size=(N_EXAMPLES_PER_TRACK), replace=False)

    for t_idx in time_indexes:
        sample = hcqt[:, t_idx : t_idx + N_TIME_FRAMES]
        label = target_salience[t_idx : t_idx + N_TIME_FRAMES, :, np.newaxis]
        yield (sample, label, t_idx)


def main(args):
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    splits = ["train", "validation"]
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

    generators = [generate_examples(split) for split in splits]

    for generator, split in zip(generators, splits):
        print(f"Generating {split} data...")
        for i, (audio, pitch_data) in enumerate(generator):
            for hcqt, target_salience, t_idx in generate_samples_from_example(audio, pitch_data):
                fname = os.path.join(data_dir, split, f"{i}-{t_idx}.npz")
                np.savez(fname, hcqt=hcqt, target_salience=target_salience)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory to save the data in")
    args = parser.parse_args()
    main(args)
