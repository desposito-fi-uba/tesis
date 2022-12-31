from statistics import mean

import numpy as np

from constants import AudioType
from utils import np_mse


class MetricsEvaluator(object):
    def __init__(
            self, tensorboard_writer, dataset, mode, filter_type, features_type, fs, generate_audios=True,
            push_to_tensorboard=True
    ):
        self.accumulative_mse_loss = []
        self.accumulative_audio_max_mse = []
        self.accumulative_audio_mse = []
        self.accumulative_samples_idx = []
        self.writer = tensorboard_writer
        self.dataset = dataset
        self.mode = mode

        self.filter_type = filter_type
        self.features_type = features_type
        self.fs = fs

        self.generate_audios = generate_audios
        self.push_to_tensorboard = push_to_tensorboard

    def restart_metrics(self):
        self.accumulative_mse_loss = []
        self.accumulative_audio_max_mse = []
        self.accumulative_audio_mse = []
        self.accumulative_samples_idx = []

    def add_metrics(self, mse_loss, samples_idx):

        self.accumulative_mse_loss.append(mse_loss.item())

        for sample_idx in samples_idx:
            if self.generate_audios:
                self.dataset.write_audio(sample_idx)

            filtered = self.dataset.get_audio(sample_idx, AudioType.FILTERED)
            noisy = self.dataset.get_audio(sample_idx, AudioType.NOISY)
            clean = self.dataset.get_audio(sample_idx, AudioType.CLEAN)

            audio_max_mse = np_mse(noisy, clean)
            audio_mse = np_mse(filtered, clean)

            self.accumulative_audio_max_mse.append(audio_max_mse)
            self.accumulative_audio_mse.append(audio_mse)

        self.accumulative_samples_idx.extend(samples_idx)

    def push_metrics(self, batches_counter):
        mse = np.asarray(self.accumulative_mse_loss).mean()
        if self.accumulative_audio_mse and self.accumulative_audio_max_mse:
            audio_mse = np.asarray(self.accumulative_audio_mse).mean()
            audio_max_mse = np.asarray(self.accumulative_audio_max_mse).mean()
        else:
            audio_mse = np.nan
            audio_max_mse = np.nan

        print(
            '[{}] {}, '
            'mse: {:.5e}, '
            'audio_max_mse: {:.5f} '
            'audio_mse: {:.5f}'.format(
                batches_counter, self.mode, mse, audio_max_mse, audio_mse
            )
        )

        if self.push_to_tensorboard:
            self.writer.add_scalar('MSE/{}'.format(self.mode), mse, batches_counter)
            if audio_max_mse is not np.nan and audio_mse is not np.nan:
                self.writer.add_scalars('Audio MSE/{}'.format(self.mode), {
                    'max': audio_max_mse,
                    'actual': audio_mse
                }, batches_counter)
            self.writer.flush()

        self.restart_metrics()
