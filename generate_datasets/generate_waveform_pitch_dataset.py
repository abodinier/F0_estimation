import os
import librosa
import mirdata
import argparse
import numpy as np

SAMPLE_RATE = 22050
N_TIME_FRAMES = 50
N_EXAMPLES_PER_TRACK = 100


def load_audio(audio_path):
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return y


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
        
        yield (y, track.pitch.frequencies)


def generate_samples_from_example(audio, pitch):
    n_times = audio.shape[0]
    all_time_indexes = np.arange(0, n_times - N_TIME_FRAMES)
    
    time_indexes = np.random.choice(all_time_indexes, size=(N_EXAMPLES_PER_TRACK), replace=False)

    for t_idx in time_indexes:
        sample = audio[t_idx : t_idx + N_TIME_FRAMES]
        label = pitch[t_idx : t_idx + N_TIME_FRAMES]
        
        yield (sample, label, t_idx)


def main(args):
    # create data save directories if they don't exist
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    splits = ["train", "validation"]
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

    # initialize split-wise generators
    generators = [generate_examples(split) for split in splits]

    # generate training and validation data
    for generator, split in zip(generators, splits):
        print(f"Generating {split} data...")
        for i, (audio, pitch_data) in enumerate(generator):
            for audio, pitch, t_idx in generate_samples_from_example(audio, pitch_data):
                fname = os.path.join(data_dir, split, f"{i}-{t_idx}.npz")
                np.savez(fname, audio=audio, pitch=pitch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args)