import numpy as np

from runsettings import RunSettings
from constants import AudioType
from utils import np_mse, pesq, stoi


class MetricsEvaluator(object):
    def __init__(
            self, tensorboard_writer, dataset, mode, filter_type, features_type, fs, generate_audios=True,
            push_to_tensorboard=True, compute_pesq_and_stoi=False, verbose_print=False
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

        self.run_settings = RunSettings()
        self.verbose_print = verbose_print

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
                pesq_value = pesq(clean, filtered, self.run_settings.fs)
                if pesq_value:
                    self.accumulative_pesq.append(pesq_value)

                min_pesq_value = pesq(clean, noisy, self.run_settings.fs)
                if min_pesq_value:
                    self.accumulative_min_pesq.append(min_pesq_value)

                stoi_value = stoi(clean, filtered, self.run_settings.fs)
                if stoi_value:
                    self.accumulative_stoi.append(stoi_value)

                min_stoi_value = stoi(clean, noisy, self.run_settings.fs)
                if min_stoi_value:
                    self.accumulative_min_stoi.append(min_stoi_value)

        self.accumulative_samples_idx.extend(samples_idx)

    def push_metrics(self, batches_counter):
        accumulative_mse_loss = np.asarray(self.accumulative_mse_loss)
        mse = accumulative_mse_loss.mean()

        if self.accumulative_audio_mse and self.accumulative_audio_max_mse:
            accumulative_audio_mse = np.asarray(self.accumulative_audio_mse)
            audio_mse_mean = accumulative_audio_mse.mean()

            accumulative_audio_max_mse = np.asarray(self.accumulative_audio_max_mse)
            audio_max_mse_mean = accumulative_audio_max_mse.mean()
        else:
            accumulative_audio_mse = np.nan
            audio_mse_mean = np.nan

            accumulative_audio_max_mse = np.nan
            audio_max_mse_mean = np.nan

        if self.accumulative_stft_mse and self.accumulative_stft_max_mse:
            accumulative_stft_mse = np.asarray(self.accumulative_stft_mse)
            stft_mse_mean = accumulative_stft_mse.mean()

            accumulative_stft_max_mse = np.asarray(self.accumulative_stft_max_mse)
            stft_max_mse_mean = accumulative_stft_max_mse.mean()
        else:
            accumulative_stft_mse = np.nan
            stft_mse_mean = np.nan

            accumulative_stft_max_mse = np.nan
            stft_max_mse_mean = np.nan

        if self.accumulative_pesq and self.accumulative_min_pesq:
            accumulative_pesq = np.asarray(self.accumulative_pesq)
            pesq_value_mean = accumulative_pesq.mean()

            accumulative_min_pesq = np.asarray(self.accumulative_min_pesq)
            min_pesq_value_mean = accumulative_min_pesq.mean()
        else:
            accumulative_pesq = np.nan
            pesq_value_mean = np.nan

            accumulative_min_pesq = np.nan
            min_pesq_value_mean = np.nan

        if self.accumulative_stoi and self.accumulative_min_stoi:
            accumulative_stoi = np.asarray(self.accumulative_stoi)
            stoi_value_mean = accumulative_stoi.mean()

            accumulative_min_stoi = np.asarray(self.accumulative_min_stoi)
            min_stoi_value_mean = accumulative_min_stoi.mean()
        else:
            accumulative_stoi = np.nan
            stoi_value_mean = np.nan

            accumulative_min_stoi = np.nan
            min_stoi_value_mean = np.nan

        print(
            '[{}] {}\n'
            '\t*mse: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*audio_max_mse: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*audio_mse: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*stft_max_mse: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*stft_mse: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*min_pesq: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*pesq {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*min_stoi: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'
            '\t*stoi: {:.5e}, min: {:.5e}, max: {:5e}, median: {:5e}\n'.format(
                batches_counter, self.mode,
                mse, np.min(accumulative_mse_loss), np.max(accumulative_mse_loss), np.median(accumulative_mse_loss),
                audio_max_mse_mean, np.min(accumulative_audio_max_mse), np.max(accumulative_audio_max_mse), np.median(accumulative_audio_max_mse),
                audio_mse_mean, np.min(accumulative_audio_mse), np.max(accumulative_audio_mse), np.median(accumulative_audio_mse),
                stft_max_mse_mean, np.min(accumulative_stft_max_mse), np.max(accumulative_stft_max_mse), np.median(accumulative_stft_max_mse),
                stft_mse_mean, np.min(accumulative_stft_mse), np.max(accumulative_stft_mse), np.median(accumulative_stft_mse),
                min_pesq_value_mean, np.min(accumulative_min_pesq), np.max(accumulative_min_pesq), np.median(accumulative_min_pesq),
                pesq_value_mean, np.min(accumulative_pesq), np.max(accumulative_pesq), np.median(accumulative_pesq),
                min_stoi_value_mean, np.min(accumulative_min_stoi), np.max(accumulative_min_stoi), np.median(accumulative_min_stoi),
                stoi_value_mean, np.min(accumulative_stoi), np.max(accumulative_stoi), np.median(accumulative_stoi),
            )
        )

        if self.push_to_tensorboard:
            self.writer.add_scalar('MSE/{}'.format(self.mode), mse, batches_counter)

            if audio_max_mse_mean is not np.nan and audio_mse_mean is not np.nan:
                self.writer.add_scalars('Audio MSE/{}'.format(self.mode), {
                    'max': audio_max_mse_mean,
                    'actual': audio_mse_mean
                }, batches_counter)

            if stft_max_mse_mean is not np.nan and stft_mse_mean is not np.nan:
                self.writer.add_scalars('STFT MSE/{}'.format(self.mode), {
                    'max': stft_max_mse_mean,
                    'actual': stft_mse_mean
                }, batches_counter)

            if pesq_value_mean is not np.nan and min_pesq_value_mean is not np.nan:
                self.writer.add_scalars('PESQ/{}'.format(self.mode), {
                    'min': min_pesq_value_mean,
                    'actual': pesq_value_mean
                }, batches_counter)

            if stoi_value_mean is not np.nan and min_stoi_value_mean is not np.nan:
                self.writer.add_scalars('STOI/{}'.format(self.mode), {
                    'min': min_stoi_value_mean,
                    'actual': stoi_value_mean
                }, batches_counter)

            self.writer.flush()

        self.restart_metrics()
