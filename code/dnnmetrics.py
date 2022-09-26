import math

import torch

from constants import FilterType, AudioType
from utils import torch_mape, remove_not_matched_snr_segments, pesq, stoi


class MetricsEvaluator(object):
    def __init__(self, tensorboard_writer, dataset, mode, filter_type, features_type, fs, compute_stoi_and_pesq=True,
                 generate_audios=True, push_to_tensorboard=True):
        self.accumulative_mse_loss = 0.0
        self.accumulative_stft_mape_loss = 0.0
        self.accumulative_stft_max_mape_loss = 0.0
        self.accumulative_time_domain_mape_loss = 0.0
        self.accumulative_time_domain_max_mape_loss = 0.0
        self.accumulative_samples_idx = []
        self.frequency_domain_amount_samples_accumulated = 0
        self.time_domain_amount_samples_accumulated = 0
        self.writer = tensorboard_writer
        self.dataset = dataset
        self.mode = mode

        self.snrs_used = ['-5', '0', '5', '10', '15', '20', 'inf']
        self.noised_used = ['babble', 'bark', 'meow', 'traffic', 'typing']

        self.stoi_threshold = {
            '-5': {
                'babble': (0.2, 0.7), 'bark': (0.2, 0.7), 'meow': (0.2, 0.7),
                'traffic': (0.2, 0.7), 'typing': (0.2, 0.7)
            },
            '0': {
                'babble': (0.2, 0.75), 'bark': (0.2, 0.75), 'meow': (0.2, 0.75),
                'traffic': (0.2, 0.75), 'typing': (0.2, 0.75)
            },
            '5': {
                'babble': (0.2, 0.8), 'bark': (0.2, 0.8), 'meow': (0.2, 0.8),
                'traffic': (0.2, 0.8), 'typing': (0.2, 0.8)
            },
            '10': {
                'babble': (0.2, 0.85), 'bark': (0.2, 0.85), 'meow': (0.2, 0.85),
                'traffic': (0.2, 0.85), 'typing': (0.2, 0.85)
            },
            '15': {
                'babble': (0.2, 0.9), 'bark': (0.2, 0.9), 'meow': (0.2, 0.9),
                'traffic': (0.2, 0.9), 'typing': (0.2, 0.9)
            },
            '20': {
                'babble': (0.2, 0.97), 'bark': (0.2, 0.97), 'meow': (0.2, 0.97),
                'traffic': (0.2, 0.97), 'typing': (0.2, 0.97)
            },
            'inf': {
                'babble': (0.2, 1), 'bark': (0.2, 1), 'meow': (0.2, 1),
                'traffic': (0.2, 1), 'typing': (0.2, 1)
            },
        }

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

        self.filter_type = filter_type
        self.features_type = features_type
        self.fs = fs

        self.sample_filtered_audio = None
        self.sample_noisy_audio = None
        self.sample_noise_audio = None
        self.sample_clean_audio = None

        self.sample_stft_clean = None
        self.sample_stft_noisy = None
        self.sample_stft_noise = None
        self.sample_stft_filtered = None
        self.sample_frequency_axis = None
        self.sample_time_axis = None

        self.compute_stoi_and_pesq = compute_stoi_and_pesq
        self.generate_audios = generate_audios
        self.push_to_tensorboard = push_to_tensorboard

    def restart_metrics(self):
        self.accumulative_mse_loss = 0.0
        self.accumulative_stft_mape_loss = 0.0
        self.accumulative_stft_max_mape_loss = 0.0
        self.accumulative_time_domain_mape_loss = 0.0
        self.accumulative_time_domain_max_mape_loss = 0.0
        self.accumulative_samples_idx = []
        self.frequency_domain_amount_samples_accumulated = 0
        self.time_domain_amount_samples_accumulated = 0
        self.accumulative_pesq = {snr: 0.0 for snr in self.snrs_used}
        self.accumulative_min_pesq = {snr: 0.0 for snr in self.snrs_used}
        self.audios_accumulated_per_snr = {snr: 0 for snr in self.snrs_used}
        self.accumulative_stoi = {snr: 0.0 for snr in self.snrs_used}
        self.accumulative_min_stoi = {snr: 0.0 for snr in self.snrs_used}
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

    def add_metrics(self, mse_loss, net_outputs, clean_speeches, noises, noisy_output_speeches,
                    samples_idx):
        stft_mape_loss = torch_mape(
            self.dataset.denormalize(net_outputs),
            self.dataset.denormalize(clean_speeches)
        )
        stft_max_mape_loss = torch_mape(
            self.dataset.denormalize(noisy_output_speeches),
            self.dataset.denormalize(clean_speeches)
        )

        self.accumulative_mse_loss += mse_loss.item()
        self.accumulative_stft_mape_loss += stft_mape_loss.item()
        self.accumulative_stft_max_mape_loss += stft_max_mape_loss.item()
        self.frequency_domain_amount_samples_accumulated += 1

        if not self.generate_audios:
            return

        for sample_idx in samples_idx:
            filtered_audio = self.dataset.build_audio(sample_idx, AudioType.FILTERED)
            noisy_audio = self.dataset.build_audio(sample_idx, AudioType.NOISY)
            noise_audio = self.dataset.build_audio(sample_idx, AudioType.NOISE)
            clean_audio = self.dataset.build_audio(sample_idx, AudioType.CLEAN)

            audio_data = self.dataset.get_audio_data(sample_idx)
            snr = str(int(audio_data['snr'])) if not math.isinf(audio_data['snr']) else 'inf'
            noise_type = audio_data['noise_type']

            self.sample_filtered_audio = filtered_audio
            self.sample_noisy_audio = noisy_audio
            self.sample_noise_audio = noise_audio
            self.sample_clean_audio = clean_audio

            if self.generate_audios:
                self.dataset.write_audio(sample_idx)

            (
                self.sample_stft_clean,
                self.sample_stft_noisy,
                _,
                self.sample_stft_noise,
                self.sample_stft_filtered,
                self.sample_frequency_axis,
                self.sample_time_axis
            ) = self.dataset.get_sample_data(sample_idx)

            time_domain_mape_loss = torch_mape(
                torch.from_numpy(filtered_audio),
                torch.from_numpy(clean_audio),
                use_median=True
            )
            time_domain_max_mape_loss = torch_mape(
                torch.from_numpy(noisy_audio),
                torch.from_numpy(clean_audio),
                use_median=True
            )

            self.accumulative_time_domain_mape_loss += time_domain_mape_loss.item()
            self.accumulative_time_domain_max_mape_loss += time_domain_max_mape_loss.item()

            self.time_domain_amount_samples_accumulated += 1

            if self.compute_stoi_and_pesq:
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

                if (min_stoi_value is not None and min_pesq_value is not None and
                        stoi_value is not None and pesq_value is not None and
                        self.stoi_threshold[snr][noise_type][0] <= min_stoi_value <=
                        self.stoi_threshold[snr][noise_type][1]):
                    self.accumulative_pesq[snr] += pesq_value
                    self.accumulative_min_pesq[snr] += min_pesq_value

                    self.accumulative_stoi[snr] += stoi_value
                    self.accumulative_min_stoi[snr] += min_stoi_value

                    self.accumulative_pesq_by_noise_type[snr][noise_type] += pesq_value
                    self.accumulative_min_pesq_by_noise_type[snr][noise_type] += min_pesq_value

                    self.accumulative_stoi_by_noise_type[snr][noise_type] += stoi_value
                    self.accumulative_min_stoi_by_noise_type[snr][noise_type] += min_stoi_value

                    self.audios_accumulated_per_snr[snr] += 1
                    self.audios_accumulated_per_snr_and_noise_type[snr][noise_type] += 1

        self.accumulative_samples_idx.extend(samples_idx)

    def push_metrics(self, batches_counter, epochs_counter):
        mse_loss_value = self.accumulative_mse_loss / self.frequency_domain_amount_samples_accumulated
        stft_mape_loss_value = self.accumulative_stft_mape_loss / self.frequency_domain_amount_samples_accumulated
        stft_max_mape_loss_value = (
                self.accumulative_stft_max_mape_loss / self.frequency_domain_amount_samples_accumulated
        )

        if self.generate_audios:
            time_domain_mape_loss_value = (
                    self.accumulative_time_domain_mape_loss / self.time_domain_amount_samples_accumulated
            )
            time_domain_max_mape_loss_value = (
                    self.accumulative_time_domain_max_mape_loss / self.time_domain_amount_samples_accumulated
            )
        else:
            time_domain_mape_loss_value = math.nan
            time_domain_max_mape_loss_value = math.nan

        if self.compute_stoi_and_pesq:
            accumulative_pesq = {snr: (
                pesq_value/self.audios_accumulated_per_snr[snr]
                if self.audios_accumulated_per_snr[snr] > 0 else math.nan
            ) for snr, pesq_value in self.accumulative_pesq.items()}

            accumulative_min_pesq = {snr: (
                pesq_value/self.audios_accumulated_per_snr[snr]
                if self.audios_accumulated_per_snr[snr] > 0 else math.nan
            ) for snr, pesq_value in self.accumulative_min_pesq.items()}

            pesq_output = ', '.join(
                ['SNR {}db: {:.5f}'.format(snr, pesq_value)
                 for snr, pesq_value in accumulative_pesq.items()]
            )
            min_pesq_output = ', '.join(
                ['SNR {}db: {:.5f}'.format(snr, pesq_value)
                 for snr, pesq_value in accumulative_min_pesq.items()]
            )

            accumulative_stoi = {snr: (
                stoi_value/self.audios_accumulated_per_snr[snr]
                if self.audios_accumulated_per_snr[snr] > 0 else math.nan
            ) for snr, stoi_value in self.accumulative_stoi.items()}

            accumulative_min_stoi = {snr: (
                stoi_value/self.audios_accumulated_per_snr[snr]
                if self.audios_accumulated_per_snr[snr] > 0 else math.nan
            ) for snr, stoi_value in self.accumulative_min_stoi.items()}

            stoi_output = ', '.join(
                ['SNR {}db: {:.5f}'.format(snr, stoi_value)
                 for snr, stoi_value in accumulative_stoi.items()]
            )
            min_stoi_output = ', '.join(
                ['SNR {}db: {:.5f}'.format(snr, stoi_value)
                 for snr, stoi_value in accumulative_min_stoi.items()]
            )

            accumulative_pesq_per_snr_and_noise_type = {
                snr: {
                    noise_type: pesq_value/self.audios_accumulated_per_snr_and_noise_type[snr][noise_type]
                    if self.audios_accumulated_per_snr_and_noise_type[snr][noise_type] > 0 else math.nan
                    for noise_type, pesq_value in pesq_by_noise_type.items()
                } for snr, pesq_by_noise_type in self.accumulative_pesq_by_noise_type.items()
            }
            accumulative_min_pesq_per_snr_and_noise_type = {
                snr: {
                    noise_type: pesq_value/self.audios_accumulated_per_snr_and_noise_type[snr][noise_type]
                    if self.audios_accumulated_per_snr_and_noise_type[snr][noise_type] > 0 else math.nan
                    for noise_type, pesq_value in pesq_by_noise_type.items()
                } for snr, pesq_by_noise_type in self.accumulative_min_pesq_by_noise_type.items()
            }

            accumulative_stoi_per_snr_and_noise_type = {
                snr: {
                    noise_type: stoi_value/self.audios_accumulated_per_snr_and_noise_type[snr][noise_type]
                    if self.audios_accumulated_per_snr_and_noise_type[snr][noise_type] > 0 else math.nan
                    for noise_type, stoi_value in stoi_by_noise_type.items()
                } for snr, stoi_by_noise_type in self.accumulative_stoi_by_noise_type.items()
            }
            accumulative_min_stoi_per_snr_and_noise_type = {
                snr: {
                    noise_type: stoi_value/self.audios_accumulated_per_snr_and_noise_type[snr][noise_type]
                    if self.audios_accumulated_per_snr_and_noise_type[snr][noise_type] > 0 else math.nan
                    for noise_type, stoi_value in stoi_by_noise_type.items()
                } for snr, stoi_by_noise_type in self.accumulative_min_stoi_by_noise_type.items()
            }

        else:
            pesq_output, min_pesq_output, stoi_output, min_stoi_output = '', '', '', ''
            accumulative_pesq, accumulative_min_pesq = {}, {}
            accumulative_stoi, accumulative_min_stoi = {}, {}
            accumulative_pesq_per_snr_and_noise_type, accumulative_min_pesq_per_snr_and_noise_type = {}, {}
            accumulative_stoi_per_snr_and_noise_type, accumulative_min_stoi_per_snr_and_noise_type = {}, {}

        print(
            '[{}] {}, '
            'mse_loss: {:.5e}, '
            'stft_mape_loss: {:.5f}, stft_max_mape_loss: {:.5f}, '
            'time_domain_mape_loss: {:.5f}, time_domain_max_mape_loss: {:.5f}, '
            'pesq: [{}], min_pesq: [{}], '
            'stoi: [{}], min_stoi: [{}], '
            'epochs {}'.format(
                batches_counter, self.mode, mse_loss_value, stft_mape_loss_value, stft_max_mape_loss_value,
                time_domain_mape_loss_value, time_domain_max_mape_loss_value, pesq_output, min_pesq_output,
                stoi_output, min_stoi_output, epochs_counter
            )
        )

        if self.push_to_tensorboard:
            # Add data to tensorboard
            self.writer.add_scalar('MSELoss/{}'.format(self.mode), mse_loss_value, batches_counter)
            self.writer.add_scalars('STFTMapeLoss/{}'.format(self.mode), {
                'max': stft_max_mape_loss_value,
                'actual': stft_mape_loss_value
            }, batches_counter)
            if self.generate_audios:
                self.writer.add_scalars('TimeDomainMapeLoss/{}'.format(self.mode), {
                    'max': time_domain_max_mape_loss_value,
                    'actual': time_domain_mape_loss_value,
                }, batches_counter)
            pesq_iterator = zip(accumulative_pesq.items(), accumulative_min_pesq.items())
            for ((snr, pesq_value), (_, min_pesq_value)) in pesq_iterator:
                if self.audios_accumulated_per_snr[snr] == 0:
                    continue
                self.writer.add_scalars('PESQ_SNR_{}db/{}'.format(snr, self.mode), {
                    'min': min_pesq_value,
                    'actual': pesq_value,
                }, batches_counter)
            stoi_iterator = zip(accumulative_stoi.items(), accumulative_min_stoi.items())
            for ((snr, stoi_value), (_, min_stoi_value)) in stoi_iterator:
                if self.audios_accumulated_per_snr[snr] == 0:
                    continue
                self.writer.add_scalars('STOI_SNR_{}db/{}'.format(snr, self.mode), {
                    'min': min_stoi_value,
                    'actual': stoi_value,
                }, batches_counter)
            pesq_by_noise_type_iterator = zip(
                accumulative_pesq_per_snr_and_noise_type.items(),
                accumulative_min_pesq_per_snr_and_noise_type.items()
            )
            for (snr, pesq_by_noise_type), (_, min_pesq_by_noise_type) in pesq_by_noise_type_iterator:
                pesq_iterator = zip(pesq_by_noise_type.items(), min_pesq_by_noise_type.items())
                for (noise_type, pesq_value), (_, min_pesq_value) in pesq_iterator:
                    if self.audios_accumulated_per_snr_and_noise_type[snr][noise_type] == 0:
                        continue
                    self.writer.add_scalars('PESQ_SNR_{}db_NOISE_{}/{}'.format(snr, noise_type, self.mode), {
                        'min': min_pesq_value,
                        'actual': pesq_value,
                    }, batches_counter)
            stoi_by_noise_type_iterator = zip(
                accumulative_stoi_per_snr_and_noise_type.items(),
                accumulative_min_stoi_per_snr_and_noise_type.items()
            )
            for (snr, stoi_by_noise_type), (_, min_stoi_by_noise_type) in stoi_by_noise_type_iterator:
                stoi_iterator = zip(stoi_by_noise_type.items(), min_stoi_by_noise_type.items())
                for (noise_type, stoi_value), (_, min_stoi_value) in stoi_iterator:
                    if self.audios_accumulated_per_snr_and_noise_type[snr][noise_type] == 0:
                        continue
                    self.writer.add_scalars('STOI_SNR_{}db_NOISE_{}/{}'.format(snr, noise_type, self.mode), {
                        'min': min_stoi_value,
                        'actual': stoi_value,
                    }, batches_counter)
            if self.generate_audios:
                if self.sample_filtered_audio is not None:
                    self.writer.add_audio(
                        'FilteredAudio/{}'.format(self.mode),
                        torch.from_numpy(self.sample_filtered_audio.reshape((-1, 1))),
                        global_step=batches_counter,
                        sample_rate=self.fs
                    )
                if self.sample_noisy_audio is not None:
                    self.writer.add_audio(
                        'NoisyAudio/{}'.format(self.mode),
                        torch.from_numpy(self.sample_noisy_audio.reshape((-1, 1))),
                        global_step=batches_counter,
                        sample_rate=self.fs
                    )
                if self.sample_noise_audio is not None:
                    self.writer.add_audio(
                        'NoiseAudio/{}'.format(self.mode),
                        torch.from_numpy(self.sample_noise_audio.reshape((-1, 1))),
                        global_step=batches_counter,
                        sample_rate=self.fs
                    )
                if self.sample_clean_audio is not None:
                    self.writer.add_audio(
                        'CleanAudio/{}'.format(self.mode),
                        torch.from_numpy(self.sample_clean_audio.reshape((-1, 1))),
                        global_step=batches_counter,
                        sample_rate=self.fs
                    )

            self.writer.flush()

        self.restart_metrics()
