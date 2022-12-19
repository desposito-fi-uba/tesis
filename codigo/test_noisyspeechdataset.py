import math
import os
from unittest import TestCase

import numpy as np
from torch.utils.data import DataLoader

from constants import AudioType
from realtimednnfilterdatasethandler import RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures


class NoisySpeechDatasetTestCase(TestCase):
    def setUp(self) -> None:
        self.fs = 16000
        self.windows_time_size = 16e-3
        self.overlap_percentage = 0.5
        self.time_feature_size = 64
        self.windows_points_size = int(2 ** np.floor(math.log2(self.windows_time_size * self.fs)))
        self.overlap_points = int(np.floor(self.windows_points_size * self.overlap_percentage))
        self.overlap_percentage = self.overlap_percentage
        self.fft_points = 2 ** 10
        self.min_audio_value = 10 ** -20
        self.log_min_audio_value = -20
        self.min_value = -25
        self.max_value = 0
        self.randomize_data = True
        self.normalize_data = True
        self.train_on_n_samples = None
        self.predict_on_time_windows = 60
        self.train_batch_size = 32

    def test_audio_reconstruction_in_real_time_dataset(self):
        dataset_path = '/home/dsesposito/Diego/Tesis/dataset/audios'
        csv_file = os.path.join(dataset_path, 'train.csv')
        dataset = RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(
            dataset_path, csv_file, self.fs, self.windows_time_size,
            self.overlap_percentage, self.fft_points, self.time_feature_size,
            randomize=self.randomize_data, max_samples=self.train_on_n_samples,
            normalize=self.normalize_data, predict_on_time_windows=self.predict_on_time_windows
        )

        data_loader = DataLoader(dataset, batch_size=self.train_batch_size)
        accumulated_samples_idx = []
        for i_batch, sample_batched in enumerate(data_loader):
            noisy_speeches = sample_batched['noisy_speech']
            samples_idx = sample_batched['sample_idx']

            end_index = self.time_feature_size - (2 * (self.time_feature_size - self.predict_on_time_windows))
            start_index = end_index - 3
            noisy_speeches = noisy_speeches[:, :, :, start_index:end_index]

            accumulated_samples_idx = dataset.accumulate_filtered_frames(noisy_speeches, samples_idx)
            if accumulated_samples_idx:
                break

        accumulated_sample_idx = accumulated_samples_idx[0]

        recovered_audio = dataset.get_audio(accumulated_sample_idx, AudioType.FILTERED)
        original_audio = dataset.get_audio(accumulated_sample_idx, AudioType.NOISY)

        self.assertTrue(np.all((np.abs(recovered_audio - original_audio)) <= 10**(-np.finfo(np.float32).precision)))
