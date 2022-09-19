import math
import warnings
from os.path import join

import matplotlib.pyplot as plt

import numpy as np
import torch

from pystoi import utils as pystoiutils
from pystoi import stoi as pystoi

from pypesq import pesq as pypesq



def torch_mape(output, target, use_median=False):
    error = torch.abs((target - output) / target)
    if use_median:
        return torch.median(error)
    else:
        return torch.mean(error)


def np_mape(output, target, use_median=False):
    error = np.abs((target - output) / target)
    if use_median:
        mape = np.median(error)
    else:
        mape = np.mean(error)
    return mape


def torch_mse(output, target):
    return torch.mean((target-output)**2)


def np_mse(output, target):
    return np.mean((target-output)**2)


def np_to_cartesian(mag_val, phase_val):
    real_val = np.multiply(
        mag_val, np.cos(phase_val))
    img_val = np.multiply(
        mag_val, np.sin(phase_val))

    return real_val + 1j * img_val


def stoi(clean_speech, noisy_speech, sample_rate):
    clean_speech = pystoiutils.resample_oct(clean_speech, 10000, sample_rate)
    noisy_speech = pystoiutils.resample_oct(noisy_speech, 10000, sample_rate)

    if len(clean_speech) <= 256 or len(noisy_speech) <= 256:
        return None

    clean_speech, noisy_speech = pystoiutils.remove_silent_frames(clean_speech, noisy_speech, 40, 256, 128)
    if len(clean_speech) <= 256 or len(noisy_speech) <= 256:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stoi_value = pystoi(clean_speech, noisy_speech, 10000)

    if stoi_value is not None and stoi_value == 1e-5:
        stoi_value = None

    return stoi_value


def pesq(clean_speech, noisy_speech, sample_rate):
    if clean_speech.shape[0] == 0 or noisy_speech.shape[0] == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pesq_value = pypesq(clean_speech, noisy_speech, sample_rate)

    if (
            pesq_value is not None and
            (
                    np.isnan(pesq_value) or
                    pesq_value == 0.0 or
                    pesq_value == 0 or
                    np.isclose(pesq_value, 0, 1e-5)
            )
    ):
        pesq_value = None

    return pesq_value


def remove_not_matched_snr_segments(clean_speech, noisy_speech, noise, enhanced_speech, snr, fs):
    window_size = 100  # in ms
    window_samples = int(fs * window_size / 1000)  # Should match with the active rms in noise + clean mixer

    audio_length = len(clean_speech)
    padded_length = math.ceil(audio_length / window_samples) * window_samples

    clean_speech = np.pad(clean_speech, [(0, padded_length - audio_length)], constant_values=np.finfo(float).eps)
    clean_speech = clean_speech.reshape((padded_length//window_samples, window_samples))
    rms_clean_speech = (clean_speech**2).mean(axis=1)**0.5

    noise = np.pad(noise, [(0, padded_length - audio_length)], constant_values=np.finfo(float).eps)
    noise[noise == 0] = np.finfo(float).eps
    noise = noise.reshape((padded_length // window_samples, window_samples))
    rms_noise = (noise ** 2).mean(axis=1) ** 0.5

    scalar_snr_low = 10**((snr - 5)/20)
    scalar_snr_sup = 10**((snr + 5)/20)
    scalar_snr = rms_clean_speech/rms_noise
    mask = np.logical_and(
        scalar_snr_low <= scalar_snr, scalar_snr <= scalar_snr_sup
    )

    noisy_speech = np.pad(noisy_speech, [(0, padded_length - audio_length)], constant_values=np.finfo(float).eps)
    noisy_speech = noisy_speech.reshape((padded_length // window_samples, window_samples))

    enhanced_speech = np.pad(enhanced_speech, [(0, padded_length - audio_length)], constant_values=np.finfo(float).eps)
    enhanced_speech = enhanced_speech.reshape((padded_length // window_samples, window_samples))

    return (
        clean_speech[mask, :].flatten(),
        noisy_speech[mask, :].flatten(),
        noise[mask, :].flatten(),
        enhanced_speech[mask, :].flatten()
    )


def frame_has_lower_snr_value(nosiy_speech, noise, snr):
    rms_noise = (noise ** 2).mean() ** 0.5
    rms_noisy = (nosiy_speech ** 2).mean() ** 0.5

    noisy_snr = rms_noisy / rms_noise
    real_snr = noisy_snr - 1
    snr = 10 ** (snr / 20)
    return real_snr <= snr


def clip_audio(audio, min_val=10**-20):
    audio[np.logical_and(-min_val < audio, audio <= 0)] = -min_val
    audio[np.logical_and(0 < audio, audio < min_val)] = min_val
    return audio
