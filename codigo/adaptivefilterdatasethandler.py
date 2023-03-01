import math
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import convolve
from torch.utils.data import IterableDataset

from constants import AudioType, AudioNotProcessed


class AdaptiveFilteringDataset(IterableDataset):
    def __init__(self, base_path, csv_file, fs, windows_time_size, overlap_percentage, max_samples=None,
                 randomize=False, constant_filter=None):
        self.base_path = base_path
        self.samples = pd.read_csv(csv_file)
        self.fs = fs
        self.windows_points_size = math.floor(windows_time_size * fs)
        self.overlap_points = int(np.floor(self.windows_points_size * overlap_percentage))
        self.overlap_percentage = overlap_percentage
        total_samples = len(self.samples)
        if max_samples is not None:
            if max_samples > total_samples:
                self.amt_samples = total_samples
            else:
                self.amt_samples = max_samples
        else:
            self.amt_samples = total_samples

        if randomize:
            self.samples_order = np.random.permutation(total_samples)[:self.amt_samples]
        else:
            self.samples_order = np.arange(self.amt_samples)
        self.current_base_path = None
        self.current_file_name = None

        self.processed_audios = OrderedDict()

        self.max_queue_size = 50

        self.noise_type = None
        self.snr = None
        self.min_audio_value = 10 ** -20
        self.constant_filter = constant_filter if constant_filter is not None else True

    def __iter__(self):
        self.frame_idx = 0
        self.sample_idx = 0
        return self

    def __next__(self):
        if self.sample_idx >= self.amt_samples:
            raise StopIteration()

        self.noise_type = self.samples.iloc[self.samples_order[self.sample_idx], 4]
        self.snr = self.samples.iloc[self.samples_order[self.sample_idx], 3]

        clean_speech_path = os.path.join(
            self.base_path, self.samples.iloc[self.samples_order[self.sample_idx], 1])
        _, clean_speech = wavfile.read(clean_speech_path)
        self.clean_speech = clean_speech

        time_shape = len(clean_speech)

        if math.isinf(self.snr):
            noise = np.zeros((time_shape,))
        else:
            noise_path = os.path.join(
                self.base_path, self.samples.iloc[self.samples_order[self.sample_idx], 2])
            _, noise = wavfile.read(noise_path)

        correlated_noise = self.build_correlated_noise(noise, clean_speech)
        self.correlated_noise = correlated_noise
        self.noise = noise

        noisy_speech_path = os.path.join(
            self.base_path, self.samples.iloc[self.samples_order[self.sample_idx], 0])
        _, noisy_speech = wavfile.read(noisy_speech_path)

        self.noisy_speech = noisy_speech

        self.current_base_path, self.current_file_name = os.path.split(clean_speech_path)

        self.processed_audios[self.sample_idx] = {
            'clean_speech': self.clean_speech,
            'noisy_speech': self.noisy_speech,
            'noise': self.noise,
            'correlated_noise': self.correlated_noise,
            'filtered_speech': [],
            'audio_name': self.get_audio_name(),
            'snr': self.snr,
            'noise_type': self.noise_type
        }
        if len(self.processed_audios) > self.max_queue_size:
            audio = next(iter(self.processed_audios))
            self.processed_audios.pop(audio)

        sample = {
            'noisy_speech': self.noisy_speech,
            'clean_speech': self.clean_speech,
            'noise': self.noise,
            'correlated_noise': self.correlated_noise,
            'audio_name': self.get_audio_name(),
            'noise_type': self.noise_type,
            'snr': self.snr,
            'sample_idx': self.sample_idx
        }
        self.sample_idx += 1
        return sample

    def get_audio_data(self, sample_idx):
        return self.processed_audios[sample_idx]

    def build_correlated_noise(self, noise, clean):
        if self.constant_filter:
            win = np.random.random((11,)) * 2 - 1
            correlated_noise = convolve(noise, win, mode='same') / sum(win)
        else:
            random_power = random.randint(14, 16)
            noise_windows_size = 2 ** random_power
            noise_size = len(noise)
            padded_size = math.ceil(noise_size / noise_windows_size) * noise_windows_size
            amount_windows = padded_size // noise_windows_size
            noise = np.pad(
                noise, [(0, padded_size - noise_size)], constant_values=0
            )
            noise = noise.reshape((amount_windows, noise_windows_size))
            correlated_noise = np.zeros((amount_windows, noise_windows_size))
            for index in range(noise.shape[0]):
                correlation_filter_length = random.randint(8, 12)
                correlation_filter = np.random.random((correlation_filter_length,))
                correlated_noise[index, :] = convolve(noise[index, :], correlation_filter, mode='same')

                cross_talk_percentage = random.randint(0, 4)
                clean_frame = clean[index * noise_windows_size:(index + 1) * noise_windows_size]
                clean_frame = np.pad(
                    clean_frame, [(0, noise_windows_size - len(clean_frame))], constant_values=0
                )
                correlated_noise[index, :] = correlated_noise[index, :] + (cross_talk_percentage / 100 * clean_frame)

            correlated_noise = correlated_noise.reshape((amount_windows*noise_windows_size))
            correlated_noise = correlated_noise[:noise_size]

        return correlated_noise

    def accumulate_filtered_speech(self, filtered_speech, sample_idx):
        self.processed_audios[sample_idx]['filtered_speech'] = filtered_speech
        return sample_idx

    def get_audio_name(self):
        file_name_without_extension, extension = os.path.splitext(self.current_file_name)
        file_name_without_extension = file_name_without_extension.replace('-clean', '')
        return file_name_without_extension

    def write_audio(self, sample_idx):
        audio = self.get_audio(sample_idx, AudioType.FILTERED)
        audio_name = self.get_audio_data(sample_idx)['audio_name']
        audio_path = os.path.join(
            self.base_path, 'filtered', 'af', f'{audio_name}.wav'
        )
        wavfile.write(audio_path, self.fs, audio)

    def get_audio(self, sample_idx, audio_type: AudioType):
        if sample_idx not in self.processed_audios:
            raise AudioNotProcessed('Audio {}, not present.'.format(sample_idx))

        audio_data = self.processed_audios[sample_idx]
        if audio_type == AudioType.NOISY:
            audio = audio_data['noisy_speech']
        elif audio_type == AudioType.NOISE:
            audio = audio_data['noise']
        elif audio_type == AudioType.CORRELATED_NOISE:
            audio = audio_data['correlated_noise']
        elif audio_type == AudioType.CLEAN:
            audio = audio_data['clean_speech']
        elif audio_type == AudioType.FILTERED:
            audio = audio_data['filtered_speech']
        else:
            raise RuntimeError('Unknown audio type {}'.format(audio_type))

        return audio

    def get_sample_data_key(self, sample_idx, key):
        if sample_idx not in self.processed_audios or key not in self.processed_audios[sample_idx]:
            return None

        return self.processed_audios[sample_idx][key]
