import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from scipy.signal import stft, istft
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co

from constants import AudioType, AudioNotProcessed
from utils import np_to_cartesian


class RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(IterableDataset):
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError('__getitem__ not implemented')

    def __init__(
            self, base_path, csv_file, fs, windows_time_size, overlap_percentage, fft_points, time_feature_size,
            device, max_samples=None, randomize=False, normalize=False, predict_on_time_windows=None,
            discard_dataset_samples_idx=None
    ):
        self.base_path = base_path
        self.samples = pd.read_csv(csv_file)
        self.data_samples = None
        self.fs = fs
        self.windows_points_size = int(2 ** np.floor(math.log2(windows_time_size * fs)))
        self.overlap_points = int(np.floor(self.windows_points_size * overlap_percentage))
        self.overlap_percentage = overlap_percentage
        self.fft_points = fft_points
        self.randomize = randomize
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

        if discard_dataset_samples_idx is not None:
            self.samples_order = np.delete(
                self.samples_order, np.in1d(self.samples_order, discard_dataset_samples_idx)
            )
            self.amt_samples = self.amt_samples - len(discard_dataset_samples_idx)

        self.current_base_path = None
        self.current_file_name = None

        self.processed_audios = OrderedDict()

        self.max_queue_size = 50

        self.min_audio_value = 10**-20
        self.min_value = -25
        self.max_value = 0
        self.normalize_io = normalize

        self.noise_type = None
        self.snr = None
        self.time_feature_size = time_feature_size
        self.predict_on_time_windows = predict_on_time_windows or self.time_feature_size
        self.device = device

    def __iter__(self):
        self.frame_idx = 0
        self.sample_idx = 0
        return self

    def __next__(self):
        if self.frame_idx == 0:
            if self.sample_idx >= self.amt_samples:
                raise StopIteration()

            noisy_speech, clean_speech, noise, audio_name, snr, noise_type = self.next_sample()
            self.init_audio(noisy_speech, clean_speech, noise, audio_name, snr, noise_type, self.sample_idx)

        stft_noisy_frame = (
            self.processed_audios[self.sample_idx]['stft_noisy_speech'][:, :, :, self.frame_idx]
        )
        stft_noisy_output_frame = (
            self.processed_audios[self.sample_idx]['stft_noisy_output_speech'][:, :, :, self.frame_idx]
        )
        stft_clean_frame = (
            self.processed_audios[self.sample_idx]['stft_clean_speech'][:, :, :, self.frame_idx]
        )
        stft_noise_frame = (
            self.processed_audios[self.sample_idx]['stft_noise'][:, :, :, self.frame_idx]
        )
        audio_name = self.processed_audios[self.sample_idx]['audio_name']
        snr = self.processed_audios[self.sample_idx]['snr']
        noise_type = self.processed_audios[self.sample_idx]['noise_type']
        sample = {
            'noisy_speech': stft_noisy_frame,
            'noisy_output_speech': stft_noisy_output_frame,
            'clean_speech': stft_clean_frame,
            'noise': stft_noise_frame,
            'audio_name': audio_name,
            'snr': snr,
            'noise_type': noise_type,
            'sample_idx': self.sample_idx
        }

        self.frame_idx += 1

        if self.frame_idx >= self.processed_audios[self.sample_idx]['amount_samples']:
            self.frame_idx = 0
            self.sample_idx += 1

        return sample

    def next_sample(self):
        noisy_speech_path = os.path.join(
            self.base_path, self.samples.iloc[self.samples_order[self.sample_idx], 0])
        _, noisy_speech = wavfile.read(noisy_speech_path)

        clean_speech_path = os.path.join(
            self.base_path, self.samples.iloc[self.samples_order[self.sample_idx], 1])
        _, clean_speech = wavfile.read(clean_speech_path)

        noise_type = self.samples.iloc[self.samples_order[self.sample_idx], 4]
        snr = self.samples.iloc[self.samples_order[self.sample_idx], 3]

        if math.isinf(snr):
            noise = np.ones(noisy_speech.shape) * self.min_audio_value
        else:
            noise_path = os.path.join(
                self.base_path, self.samples.iloc[self.samples_order[self.sample_idx], 2])
            _, noise = wavfile.read(noise_path)

        audio_name = self.get_audio_name(clean_speech_path)
        return noisy_speech, clean_speech, noise, audio_name, snr, noise_type

    def forward_to_noise_type_and_snr(self, noise_type, snr, skip_amount):
        found = 0
        snr = float(snr)
        while True:
            _, _, _, _, curr_snr, curr_noise_type = self.next_sample()
            if curr_noise_type != noise_type or curr_snr != snr:
                self.sample_idx += 1
                continue
            found += 1
            if found <= skip_amount:
                self.sample_idx += 1
                continue

            break

    def init_audio(self, noisy_speech, clean_speech, noise, audio_name, snr, noise_type, audio_index):
        not_padded_time_shape = len(noisy_speech)

        stft_noisy_speech, stft_phase_noisy_speech, _ = self.prepare_input(noisy_speech)
        stft_noisy_output_speech, noisy_speech = self.prepare_output(noisy_speech)
        stft_noisy_speech = stft_noisy_speech.to(self.device)

        stft_clean_speech, clean_speech = self.prepare_output(clean_speech)
        stft_clean_speech = stft_clean_speech.to(self.device)

        stft_noise, noise = self.prepare_output(noise)

        amount_frames = stft_noisy_speech.shape[3]

        self.processed_audios[audio_index] = {
            'stft_clean_speech': stft_clean_speech,
            'stft_noisy_speech': stft_noisy_speech,
            'stft_noisy_output_speech': stft_noisy_output_speech,
            'stft_noise': stft_noise,
            'stft_phase_noisy_speech': stft_phase_noisy_speech,
            'stft_filtered_speech': [],
            'clean_audio': clean_speech,
            'noisy_audio': noisy_speech,
            'noise_audio': noise,
            'audio_name': audio_name,
            'amount_samples': amount_frames,
            'not_padded_time_shape': not_padded_time_shape,
            'snr': snr,
            'noise_type': noise_type
        }
        if len(self.processed_audios) > self.max_queue_size:
            audio = next(iter(self.processed_audios))
            self.processed_audios.pop(audio)

    def get_stft(self, audio):
        _, _, audio_complex_stft = stft(
            audio, fs=self.fs, nperseg=self.windows_points_size, noverlap=self.overlap_points,
            nfft=self.fft_points - 1, return_onesided=True
        )
        audio_magnitude_stft = np.abs(audio_complex_stft)
        audio_magnitude_stft[audio_magnitude_stft < self.min_audio_value] = self.min_audio_value
        audio_magnitude_stft = np.log10(audio_magnitude_stft).astype(np.float32)
        audio_magnitude_stft = self.normalize(audio_magnitude_stft)
        return audio_magnitude_stft

    def crop_audio(self, audio):
        mask = np.logical_and(audio >= 0, audio < self.min_audio_value)
        audio[mask] = self.min_audio_value

        mask = np.logical_and(audio < 0, audio > -self.min_audio_value)
        audio[mask] = -self.min_audio_value

        return audio

    def prepare_input(self, audio):
        audio_size = len(audio)
        frame_size = (self.time_feature_size - 1) * self.overlap_points
        amount_frames = (math.ceil((audio_size - frame_size) / self.windows_points_size)) + 1
        padded_audio_size = (amount_frames - 1) * self.windows_points_size + frame_size
        audio = np.pad(
            audio, [(0, padded_audio_size - audio_size)],
            constant_values=self.min_audio_value
        )

        audio = self.crop_audio(audio)

        audio_frames_complex_stft = np.zeros((self.fft_points//2, self.time_feature_size, amount_frames), dtype=complex)
        _, _, audio_complex_stft = stft(
            audio, fs=self.fs, nperseg=self.windows_points_size, noverlap=self.overlap_points,
            nfft=self.fft_points - 1, return_onesided=True
        )

        for i in range(0, amount_frames):
            audio_frames_complex_stft[:, :, i] = audio_complex_stft[:, i*2:self.time_feature_size+(i*2)]

        audio_frames_magnitude_stft = np.abs(audio_frames_complex_stft)
        # audio_frames_magnitude_stft[audio_frames_magnitude_stft < self.min_audio_value] = self.min_audio_value
        audio_frames_magnitude_stft = np.log10(audio_frames_magnitude_stft).astype(np.float32)
        audio_frames_magnitude_stft = self.normalize(audio_frames_magnitude_stft)
        audio_frames_magnitude_stft = np.expand_dims(audio_frames_magnitude_stft, axis=0)

        audio_frames_phase_stft = np.angle(audio_frames_complex_stft)
        return (
            torch.from_numpy(audio_frames_magnitude_stft),
            torch.from_numpy(audio_frames_phase_stft),
            torch.from_numpy(audio[:audio_size])
        )

    def prepare_output(self, audio):
        audio_size = len(audio)
        frame_size = (self.time_feature_size - 1) * self.overlap_points
        amount_frames = (math.ceil((audio_size - frame_size) / self.windows_points_size)) + 1
        padded_audio_size = (amount_frames - 1) * self.windows_points_size + frame_size
        audio = np.pad(
            audio, [(0, padded_audio_size - audio_size)],
            constant_values=self.min_audio_value
        )

        audio = self.crop_audio(audio)

        audio_frames_complex_stft = np.zeros((self.fft_points // 2, 3, amount_frames), dtype=complex)
        _, _, audio_complex_stft = stft(
            audio, fs=self.fs, nperseg=self.windows_points_size, noverlap=self.overlap_points,
            nfft=self.fft_points - 1, return_onesided=True
        )
        for i in range(0, amount_frames):
            audio_frame_complex_stft = audio_complex_stft[:, i * 2:self.time_feature_size + (i * 2)]

            end_index = self.time_feature_size - (2 * (self.time_feature_size - self.predict_on_time_windows))
            start_index = end_index - 3

            audio_frames_complex_stft[:, :, i] = audio_frame_complex_stft[:, start_index:end_index]

        audio_frames_magnitude_stft = np.abs(audio_frames_complex_stft)
        # audio_frames_magnitude_stft[audio_frames_magnitude_stft < self.min_audio_value] = self.min_audio_value
        audio_frames_magnitude_stft = np.log10(audio_frames_magnitude_stft).astype(np.float32)
        audio_frames_magnitude_stft = self.normalize(audio_frames_magnitude_stft)
        audio_frames_magnitude_stft = np.expand_dims(audio_frames_magnitude_stft, axis=0)

        return torch.from_numpy(audio_frames_magnitude_stft), torch.from_numpy(audio[:audio_size])

    def get_audio(self, sample_idx, audio_type: AudioType):
        if sample_idx not in self.processed_audios:
            raise AudioNotProcessed('Audio {}, not present.'.format(sample_idx))

        audio_data = self.processed_audios[sample_idx]
        if audio_type == AudioType.NOISY:
            return audio_data['noisy_audio'].numpy()
        elif audio_type == AudioType.NOISE:
            return audio_data['noise_audio'].numpy()
        elif audio_type == AudioType.CLEAN:
            return audio_data['clean_audio'].numpy()
        elif audio_type == AudioType.FILTERED:
            stft_frames_magnitude, stft_frames_phase = (
                audio_data['stft_filtered_speech'], audio_data['stft_phase_noisy_speech']
            )
        else:
            raise RuntimeError('Unknown audio type {}'.format(audio_type))

        audio_size = audio_data['not_padded_time_shape']
        frame_size = (self.time_feature_size - 1) * self.overlap_points
        amount_frames = stft_frames_magnitude.shape[3]
        padded_audio_size = (amount_frames - 1) * self.windows_points_size + frame_size

        stft_frames_magnitude = self.denormalize(stft_frames_magnitude)
        stft_frames_magnitude = 10 ** stft_frames_magnitude

        audio = np.zeros((padded_audio_size,))
        audio[:audio_size] = audio_data['noisy_audio'].numpy().copy()
        offset = frame_size - self.windows_points_size * (self.time_feature_size - self.predict_on_time_windows + 1)
        for i in range(0, amount_frames):
            end_index = self.time_feature_size - (2 * (self.time_feature_size - self.predict_on_time_windows))
            start_index = end_index - 3
            frame_phase = stft_frames_phase[:, start_index:end_index, i]
            stft_frame_complex = np_to_cartesian(stft_frames_magnitude[0, :, :, i], frame_phase)
            _, audio_window = istft(
                stft_frame_complex, fs=self.fs, nperseg=self.windows_points_size, noverlap=self.overlap_points,
                nfft=self.fft_points - 1, input_onesided=True
            )

            start_index = offset + i*self.windows_points_size
            end_index = offset + (i+1)*self.windows_points_size
            audio[start_index: end_index] = audio_window

        audio = audio[:audio_size]
        return audio

    def write_audio(self, sample_idx):
        audio = self.get_audio(sample_idx, AudioType.FILTERED)
        audio_name = self.get_audio_data(sample_idx)['audio_name']
        audio_directory = os.path.join(
            self.base_path, '..', 'filtered', 'dnn'
        )
        Path(audio_directory).mkdir(parents=True, exist_ok=True)

        audio_path = os.path.join(
            audio_directory, f'{audio_name}.wav'
        )
        wavfile.write(audio_path, self.fs, audio)

    def accumulate_filtered_frames(self, filtered_frames, samples_idx):
        accumulated_samples_idx = []
        for i in range(filtered_frames.shape[0]):
            accumulated_sample_idx = self.accumulate_filtered_frame(
                filtered_frames[i, :, :, :].numpy(), samples_idx[i].item()
            )
            if accumulated_sample_idx is not None:
                accumulated_samples_idx.append(accumulated_sample_idx)

        return accumulated_samples_idx

    def accumulate_filtered_frame(self, filtered_frame, sample_idx):
        if sample_idx not in self.processed_audios:
            raise RuntimeError(
                f'Sample {sample_idx} was deleted from memory. '
                f'Samples indexes in memory are {list(self.processed_audios.keys())}'
            )
        amount_samples = self.processed_audios[sample_idx]['amount_samples']
        stft_filtered_speech = self.processed_audios[sample_idx].setdefault('stft_filtered_speech', [])
        stft_filtered_speech.append(filtered_frame)
        if len(stft_filtered_speech) == amount_samples:
            frames = np.asarray(stft_filtered_speech)
            frames = frames.reshape(
                (frames.shape[0], -1)
            ).swapaxes(
                0, 1
            ).reshape(
                frames.shape[1:] + (frames.shape[0], )
            )
            self.processed_audios[sample_idx]['stft_filtered_speech'] = frames
            return sample_idx
        elif len(stft_filtered_speech) < amount_samples:
            return None
        else:
            raise RuntimeError('Something went wrong')

    def get_sample_data(self, sample_idx):
        if sample_idx not in self.processed_audios:
            return None

        audio_data = self.processed_audios[sample_idx]
        stft_noisy_output = audio_data['stft_noisy_output_speech'][0, :, :, :]
        stft_noisy = audio_data['stft_noisy_speech'][0, :, :, :]
        stft_clean = audio_data['stft_clean_speech'][0, :, :, :]
        stft_noise = audio_data['stft_noise'][0, :, :, :]
        stft_filtered = audio_data['stft_filtered_speech'][0, :, :, :]
        return (
            self.denormalize(stft_clean.cpu().numpy()),
            self.denormalize(stft_noisy_output.cpu().numpy()),
            self.denormalize(stft_noisy.cpu().numpy()),
            self.denormalize(stft_noise.cpu().numpy()),
            self.denormalize(stft_filtered),
            np.arange(0, stft_filtered.shape[0]),
            np.arange(0, stft_filtered.shape[1]) * self.windows_points_size / self.fs
        )

    def get_frq_time_axis(self, freq_shape, time_shape):
        return (
            np.arange(0, freq_shape),
            np.arange(0, time_shape) * self.windows_points_size / self.fs
        )

    def get_sample_data_key(self, sample_idx, key):
        if sample_idx not in self.processed_audios or key not in self.processed_audios[sample_idx]:
            return None

        return self.denormalize(self.processed_audios[sample_idx][key])

    @classmethod
    def get_audio_name(cls, noisy_speech_path):
        _, file_name = os.path.split(noisy_speech_path)
        file_name_without_extension, extension = os.path.splitext(file_name)
        file_name_without_extension = file_name_without_extension.replace('-clean', '')
        return file_name_without_extension

    def is_new_sample(self):
        return self.frame_idx == 0

    def normalize(self, frame):
        if not self.normalize_io:
            return frame

        return (frame - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, frame):
        if not self.normalize_io:
            return frame

        return frame * (self.max_value - self.min_value) + self.min_value

    def get_audio_data(self, sample_idx):
        return self.processed_audios[sample_idx]
