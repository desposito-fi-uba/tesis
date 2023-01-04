from statistics import mean

import numpy as np

import runsettings
from constants import AudioType
from utils import np_mse, pesq, stoi


class MetricsEvaluator(object):
    def __init__(
            self, tensorboard_writer, dataset, mode, filter_type, features_type, fs, generate_audios=True,
            push_to_tensorboard=True, compute_pesq_and_stoi=False
    ):
        self.accumulative_mse_loss = []
        self.accumulative_audio_max_mse = []
        self.accumulative_audio_mse = []
        self.accumulative_samples_idx = []
        self.accumulative_stft_max_mse = []
        self.accumulative_stft_mse = []
        self.accumulative_pesq = []
        self.accumulative_min_pesq = []
        self.accumulative_stoi = []
        self.accumulative_min_stoi = []
        self.writer = tensorboard_writer
        self.dataset = dataset
        self.mode = mode

        self.filter_type = filter_type
        self.features_type = features_type
        self.fs = fs

        self.generate_audios = generate_audios
        self.push_to_tensorboard = push_to_tensorboard
        self.compute_pesq_and_stoi = compute_pesq_and_stoi

    def restart_metrics(self):
        self.accumulative_mse_loss = []
        self.accumulative_audio_max_mse = []
        self.accumulative_audio_mse = []
        self.accumulative_samples_idx = []
        self.accumulative_stft_max_mse = []
        self.accumulative_stft_mse = []
        self.accumulative_pesq = []
        self.accumulative_min_pesq = []
        self.accumulative_stoi = []
        self.accumulative_min_stoi = []

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

            clean_stft, noisy_stft, _, _, filtered_stft, _, _ = self.dataset.get_sample_data(sample_idx)

            stft_max_mse = np_mse(noisy_stft, clean_stft)
            stft_mse = np_mse(filtered_stft, clean_stft)

            self.accumulative_stft_max_mse.append(stft_max_mse)
            self.accumulative_stft_mse.append(stft_mse)

            if self.compute_pesq_and_stoi:
                pesq_value = pesq(clean, filtered, runsettings.fs)
                if pesq_value:
                    self.accumulative_pesq.append(pesq_value)

                min_pesq_value = pesq(clean, noisy, runsettings.fs)
                if min_pesq_value:
                    self.accumulative_min_pesq.append(min_pesq_value)

                stoi_value = stoi(clean, filtered, runsettings.fs)
                if stoi_value:
                    self.accumulative_stoi.append(stoi_value)

                min_stoi_value = stoi(clean, noisy, runsettings.fs)
                if min_stoi_value:
                    self.accumulative_min_stoi.append(min_stoi_value)

        self.accumulative_samples_idx.extend(samples_idx)

    def push_metrics(self, batches_counter):
        mse = np.asarray(self.accumulative_mse_loss).mean()
        if self.accumulative_audio_mse and self.accumulative_audio_max_mse:
            audio_mse = np.asarray(self.accumulative_audio_mse).mean()
            audio_max_mse = np.asarray(self.accumulative_audio_max_mse).mean()
        else:
            audio_mse = np.nan
            audio_max_mse = np.nan

        if self.accumulative_stft_mse and self.accumulative_stft_max_mse:
            stft_mse = np.asarray(self.accumulative_stft_mse).mean()
            stft_max_mse = np.asarray(self.accumulative_stft_max_mse).mean()
        else:
            stft_mse = np.nan
            stft_max_mse = np.nan

        if self.accumulative_pesq and self.accumulative_min_pesq:
            pesq_value = np.asarray(self.accumulative_pesq).mean()
            min_pesq_value = np.asarray(self.accumulative_min_pesq).mean()
        else:
            pesq_value = np.nan
            min_pesq_value = np.nan

        if self.accumulative_stoi and self.accumulative_min_stoi:
            stoi_value = np.asarray(self.accumulative_stoi).mean()
            min_stoi_value = np.asarray(self.accumulative_min_stoi).mean()
        else:
            stoi_value = np.nan
            min_stoi_value = np.nan

        print(
            '[{}] {}, '
            'mse: {:.5e}, '
            'audio_max_mse: {:.5e} '
            'audio_mse: {:.5e} '
            'stft_max_mse: {:.5e} '
            'stft_mse: {:.5e}'.format(
                batches_counter, self.mode, mse, audio_max_mse, audio_mse, stft_max_mse, stft_mse
            )
        )

        if self.push_to_tensorboard:
            self.writer.add_scalar('MSE/{}'.format(self.mode), mse, batches_counter)

            if audio_max_mse is not np.nan and audio_mse is not np.nan:
                self.writer.add_scalars('Audio MSE/{}'.format(self.mode), {
                    'max': audio_max_mse,
                    'actual': audio_mse
                }, batches_counter)

            if stft_max_mse is not np.nan and stft_mse is not np.nan:
                self.writer.add_scalars('STFT MSE/{}'.format(self.mode), {
                    'max': stft_max_mse,
                    'actual': stft_mse
                }, batches_counter)

            if pesq_value is not np.nan and min_pesq_value is not np.nan:
                self.writer.add_scalars('PESQ/{}'.format(self.mode), {
                    'min': min_pesq_value,
                    'actual': pesq_value
                }, batches_counter)

            if stoi_value is not np.nan and min_stoi_value is not np.nan:
                self.writer.add_scalars('STOI/{}'.format(self.mode), {
                    'min': min_stoi_value,
                    'actual': stoi_value
                }, batches_counter)

            self.writer.flush()

        self.restart_metrics()
