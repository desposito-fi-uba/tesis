import math

import torch

import runsettings
from constants import AudioType
from utils import np_mse, np_mape, remove_not_matched_snr_segments, stoi, pesq


class AdaptiveFilteringMetricsEvaluator(object):
    def __init__(self, tensorboard_writer, dataset, mode, fs):
        self.accumulative_frame_mse = 0.0
        self.accumulative_audio_mse = 0.0
        self.accumulative_audio_max_mse = 0.0
        self.accumulative_samples_idx = []
        self.audios_accumulated = 0
        self.frames_accumulated = 0
        self.writer = tensorboard_writer
        self.dataset = dataset
        self.mode = mode

        self.snrs_used = ['-5', '0', '5', '10', '15', '20', 'inf']
        self.noised_used = ['babble', 'bark', 'meow', 'traffic', 'typing']

        self.accumulative_pesq_by_noise_type = {
            snr: {noise_type: 0.0 for noise_type in self.noised_used} for snr in self.snrs_used
        }
        self.accumulative_min_pesq_by_noise_type = {
            snr: {noise_type: 0.0 for noise_type in self.noised_used} for snr in self.snrs_used
        }

        self.accumulative_stoi_by_noise_type = {
            snr: {noise_type: 0.0 for noise_type in self.noised_used} for snr in self.snrs_used
        }
        self.accumulative_min_stoi_by_noise_type = {
            snr: {noise_type: 0.0 for noise_type in self.noised_used} for snr in self.snrs_used
        }

        self.audios_accumulated_per_snr_and_noise_type = {
            snr: {noise_type: 0 for noise_type in self.noised_used} for snr in self.snrs_used
        }

        self.accumulative_pesq = {snr: 0.0 for snr in self.snrs_used}
        self.accumulative_min_pesq = {snr: 0.0 for snr in self.snrs_used}

        self.accumulative_stoi = {snr: 0.0 for snr in self.snrs_used}
        self.accumulative_min_stoi = {snr: 0.0 for snr in self.snrs_used}

        self.audios_accumulated_per_snr = {snr: 0 for snr in self.snrs_used}

        self.fs = fs

        self.sample_filtered_audio = None
        self.sample_noisy_audio = None
        self.sample_noise_audio = None
        self.sample_clean_audio = None

    def save_metrics(self, sample_idx):
        if runsettings.test_save_filtered_audios:
            self.dataset.write_audio(sample_idx)
            return

        audio_data = self.dataset.get_audio_data(sample_idx)
        snr = str(int(audio_data['snr'])) if not math.isinf(audio_data['snr']) else 'inf'
        noise_type = audio_data['noise_type']

        filtered_audio = self.dataset.get_audio(sample_idx, AudioType.FILTERED)
        noisy_audio = self.dataset.get_audio(sample_idx, AudioType.NOISY)
        noise_audio = self.dataset.get_audio(sample_idx, AudioType.NOISE)
        clean_audio = self.dataset.get_audio(sample_idx, AudioType.CLEAN)

        audio_mse = np_mse(
            filtered_audio,
            clean_audio
        )
        audio_max_mse = np_mse(
            noisy_audio,
            clean_audio
        )

        self.writer.add_scalars('MSE_SNR_{}db/{}'.format(snr, self.mode), {
            'max': audio_max_mse,
            'actual': audio_mse,
        }, sample_idx)
        self.writer.add_scalars('MSE_SNR_{}db_NOISE_{}/{}'.format(snr, noise_type, self.mode), {
            'max': audio_max_mse,
            'actual': audio_mse,
        }, sample_idx)

        if snr != 'inf':
            clean_audio, noisy_audio, noise_audio, filtered_audio = remove_not_matched_snr_segments(
                clean_audio, noisy_audio, noise_audio, filtered_audio, int(snr), self.fs
            )
            min_stoi_value = stoi(clean_audio, noisy_audio, self.fs)
            min_pesq_value = pesq(clean_audio, noisy_audio, self.fs)
        else:
            min_stoi_value = 1.0
            min_pesq_value = 4.5

        stoi_value = stoi(clean_audio, filtered_audio, self.fs)
        pesq_value = pesq(clean_audio, filtered_audio, self.fs)

        self.writer.add_scalars('AudioMSELoss/{}'.format(self.mode), {
            'max': audio_mse,
            'actual': audio_max_mse,
        }, sample_idx)
        if pesq_value is not None:
            self.writer.add_scalars('PESQ_SNR_{}db/{}'.format(snr, self.mode), {
                'min': min_pesq_value,
                'actual': pesq_value,
            }, sample_idx)
            self.writer.add_scalars('PESQ_SNR_{}db_NOISE_{}/{}'.format(snr, noise_type, self.mode), {
                'min': min_pesq_value,
                'actual': pesq_value,
            }, sample_idx)

        if stoi_value is not None:
            self.writer.add_scalars('STOI_SNR_{}db/{}'.format(snr, self.mode), {
                'min': min_stoi_value,
                'actual': stoi_value,
            }, sample_idx)
            self.writer.add_scalars('STOI_SNR_{}db_NOISE_{}/{}'.format(snr, noise_type, self.mode), {
                'min': min_stoi_value,
                'actual': stoi_value,
            }, sample_idx)

        if runsettings.test_generate_audios:
            self.writer.add_audio(
                'FilteredAudio/{}'.format(self.mode),
                torch.from_numpy(filtered_audio.reshape((-1, 1))),
                global_step=sample_idx,
                sample_rate=self.fs
            )
            self.writer.add_audio(
                'NoisyAudio/{}'.format(self.mode),
                torch.from_numpy(noisy_audio.reshape((-1, 1))),
                global_step=sample_idx,
                sample_rate=self.fs
            )
            self.writer.add_audio(
                'NoiseAudio/{}'.format(self.mode),
                torch.from_numpy(noise_audio.reshape((-1, 1))),
                global_step=sample_idx,
                sample_rate=self.fs
            )
            self.writer.add_audio(
                'CleanAudio/{}'.format(self.mode),
                torch.from_numpy(clean_audio.reshape((-1, 1))),
                global_step=sample_idx,
                sample_rate=self.fs
            )
