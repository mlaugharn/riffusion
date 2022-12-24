"""
convert a wav to a bunch of new seed images

"""
import pathlib

import PIL.Image
import numpy as np
import librosa

from riffusion.audio import spectrogram_from_waveform, image_from_spectrogram, wav_bytes_from_spectrogram_image


def waveform_from_fname(fname: str, use_original_sr=False) -> np.ndarray:
    """
    Load a WAV file into a waveform.
    """
    y, sr = librosa.load(fname, sr=None if use_original_sr else 44100)
    if use_original_sr:
        return y, sr
    else:
        return y, 44100


def get_spectrogram_slices_from_fname(fname, slice_length: float, overlap: float, *wavform_args, **wavform_kwargs):
    """
    Get a list of spectrogram images (in PIL Image format) from a WAV file.
    The slice length and overlap is in seconds.
    """

    waveform, sample_rate = waveform_from_fname(fname, use_original_sr=False)
    waveform = waveform.astype(np.float32)

    slice_size = int(slice_length * sample_rate)
    overlap_size = int(overlap * sample_rate)

    spectrogram_slices = []
    start = 0
    while start + slice_size < len(waveform):
        end = start + slice_size
        slice_waveform = waveform[start:end]
        spectrogram = spectrogram_from_waveform(slice_waveform, sample_rate, *wavform_args, **wavform_kwargs)
        spectrogram_slices.append(spectrogram)
        start += slice_size - overlap_size

    return spectrogram_slices


def default_behavior(fname='../audio_samples/silver_b.mp3'):
    """
    """
    fname = str(pathlib.Path(fname).absolute())
    stem = pathlib.Path(fname).stem
    slice_length = 10
    overlap = 0

    sample_rate = 44100  # [Hz]
    clip_duration_ms = slice_length * 1000  # [ms]

    bins_per_image = 512
    n_mels = 512

    # FFT parameters
    window_duration_ms = 100  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 10  # [ms]

    # Derived parameters
    image_width = 512 * 2
    num_samples = int(image_width / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    spectrogram_slices = get_spectrogram_slices_from_fname(fname, slice_length, overlap, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mels=n_mels)
    for i, slice_ in enumerate(spectrogram_slices):
        slice_fname = pathlib.Path(f"../seed_images/{stem}_{i}.png").absolute()
        image = image_from_spectrogram(slice_)
        image.save(slice_fname)


def test_params_from_example():
    img = "../seed_images/og_beat.png"
    img = PIL.Image.open(img).convert('RGB')
    wav_bytes, duration_s = wav_bytes_from_spectrogram_image(img)


"""
load filename from args
"""
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='../audio_samples/silver_b.mp3')
    args = parser.parse_args()
    default_behavior(args.fname)