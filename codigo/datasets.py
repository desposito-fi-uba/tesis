import csv
import glob
import os
import random
from os.path import isfile, join

import peakutils
from numpy.fft import fftfreq

import matplotlib.pyplot as plt
import click
import librosa
import numpy as np
from scipy.fft import fft
from scipy.io import wavfile

from audiolib import segmental_snr_mixer
from utils import stoi, pesq, remove_not_matched_snr_segments


@click.group()
def entry_point():
    pass


@entry_point.command()
@click.option('--max-amount', default=0, required=False, type=int)
@click.option('--env', default='train', required=False, type=str)
def generate(max_amount, env):
    if env != 'train' and env != 'test':
        raise RuntimeError(f'Wrong env {env}. Options are train or test')

    dataset_path = join(os.getcwd(), 'raw-dataset', f'clean_{env}')
    speeches_audio_path = [
        join(dataset_path, file_path) for file_path in os.listdir(dataset_path) if isfile(join(dataset_path, file_path))
    ]
    # Using twice each speech audio for augmentation
    audios = np.array(speeches_audio_path + speeches_audio_path)
    np.random.shuffle(audios)

    dataset_path = join(os.getcwd(), 'raw-dataset', f'noise_{env}')
    noises = np.asarray(
        [file_path for file_path in glob.glob(f'{dataset_path}/*.wav', recursive=True)]
    )
    np.random.shuffle(noises)

    if max_amount > 0:
        audios = audios[:max_amount]

    dataset_path = join(os.getcwd(), 'dataset')
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    audios_path = join(dataset_path, f'audios_{env}')

    if not os.path.isdir(audios_path):
        os.mkdir(audios_path)

    rows = []
    noise_index = 0
    amount_noise_used = 0
    noise = None
    noise_type = None
    sr = 16000
    silent = 0.2
    silent_samples = int(sr*silent)
    continue_generating = True
    print('Starting mixing {} clean speeches with {} noises'.format(len(audios), len(noises)))
    for index, audio_path in enumerate(audios):
        if not continue_generating:
            break

        _, file_name = os.path.split(audio_path)
        file_name_without_extension, extension = os.path.splitext(file_name)
        try:
            clean_speech, _ = librosa.load(audio_path, sr=sr)
        except Exception:
            print('Error reading file {}. Continuing with the next audio'.format(audio_path))
            continue

        amount_speech_samples = len(clean_speech)
        noise_to_mix = np.zeros((silent_samples + amount_speech_samples, ))

        snr_choices = [-5, 0, 5, 10, 15, 20, -5, 0, 5, 10, 15, 20, np.inf]
        snr = snr_choices[np.random.randint(len(snr_choices))]

        if snr != np.inf:
            if amount_noise_used == 0:
                if noise_index + 1 > len(noises):
                    np.random.shuffle(noises)
                    print('Run out of noise data after processing {}'.format(index+1))
                    noise_index = 0

                continue_reading_noises = True
                while continue_reading_noises:
                    try:
                        noise, _ = librosa.load(noises[noise_index], sr=sr)
                        noise_type = os.path.basename(noises[noise_index]).split('_')[0]
                        continue_reading_noises = False
                        break
                    except Exception:
                        print('Error reading noise {}. Continuing with the next audio'.format(audio_path))
                        noise_index += 1
                        continue

            available_noise = noise[amount_noise_used:]
            amount_available_noise = len(available_noise)
            if amount_available_noise > amount_speech_samples:
                noise_to_mix[silent_samples:] = available_noise[:amount_speech_samples]
                amount_noise_used = amount_noise_used + amount_speech_samples - 1
            elif amount_available_noise == amount_speech_samples:
                noise_to_mix[silent_samples:] = available_noise
                amount_noise_used = 0
                noise_index += 1
            else:
                noise_to_mix[silent_samples:amount_available_noise+silent_samples] = available_noise
                amount_noise_used = 0
                noise_index += 1

            clean_new_level, noise_new_level, noisy_speech, _ = segmental_snr_mixer(clean_speech, noise_to_mix, snr)

            clean_path_with_out_base_path = os.path.join(
                '{}-{}-clean{}'.format(index, file_name_without_extension, extension))
            noisy_path_with_out_base_path = os.path.join(
                '{}-{}-noisy{}'.format(index, file_name_without_extension, extension))
            noise_path_with_out_base_path = os.path.join(
                '{}-{}-noise{}'.format(index, file_name_without_extension, extension))
            clean_path = join(audios_path, clean_path_with_out_base_path)
            noisy_path = join(audios_path, noisy_path_with_out_base_path)
            noise_path = join(audios_path, noise_path_with_out_base_path)
            wavfile.write(clean_path, sr, clean_new_level)
            wavfile.write(noisy_path, sr, noisy_speech)
            wavfile.write(noise_path, sr, noise_new_level)

            rows.append([
                noisy_path_with_out_base_path, clean_path_with_out_base_path, noise_path_with_out_base_path, snr,
                noise_type
            ])
        else:
            clean_new_level = clean_speech

            clean_path_with_out_base_path = os.path.join(
                '{}-{}-clean{}'.format(index, file_name_without_extension, extension))
            clean_path = join(audios_path, clean_path_with_out_base_path)
            wavfile.write(clean_path, sr, clean_new_level)

            rows.append([clean_path_with_out_base_path, clean_path_with_out_base_path, None, snr, noise_type])

    random.shuffle(rows)

    csv_path = join(audios_path, f'{env}.csv')
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            writer.writerow(row)


@entry_point.command()
@click.option('--audios-input-dir', default='', required=False, type=str)
def compute_pesq_stoi_histogram(audios_input_dir):
    if not audios_input_dir:
        mixed_path = join(os.getcwd(), '..', 'dataset', 'audios')
    else:
        mixed_path = join(os.getcwd(), '..', 'dataset', audios_input_dir)

    rows = []

    csv_path = join(mixed_path, 'test.csv')
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows.extend([row for row in reader])

    csv_path = join(mixed_path, 'train.csv')
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows.extend([row for row in reader])

    csv_path = join(mixed_path, 'val.csv')
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows.extend([row for row in reader])

    amount_stoi_values_ignored = 0
    amount_pesq_values_ignored = 0
    amount_audios = len(rows)

    snrs_used = ['-5', '0', '5', '10', '15', '20', 'inf']
    noised_used = ['babble', 'bark', 'meow', 'traffic', 'typing']
    pesq_values_per_snr_and_noise_type = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }
    stoi_values_per_snr_and_noise_type = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }
    for index, row in enumerate(rows):
        (
            noisy_path_with_out_base_path, clean_path_with_out_base_path, noise_path_with_out_base_path,
            snr, noise_type
        ) = tuple(row)

        if snr == 'inf':
            stoi_value = 1.0
            pesq_value = 4.5
            pesq_values_per_snr_and_noise_type[snr][noise_type].append(pesq_value)
            stoi_values_per_snr_and_noise_type[snr][noise_type].append(stoi_value)
            continue

        clean_speech, sample_rate = librosa.load(join(mixed_path, clean_path_with_out_base_path), sr=None)
        noisy_speech, sample_rate = librosa.load(join(mixed_path, noisy_path_with_out_base_path), sr=None)
        noise, sample_rate = librosa.load(join(mixed_path, noise_path_with_out_base_path), sr=None)

        clean_speech, noisy_speech, noise, _ = remove_not_matched_snr_segments(
            clean_speech, noisy_speech, noise, np.zeros(clean_speech.shape), int(snr), sample_rate
        )

        stoi_value = stoi(clean_speech, noisy_speech, sample_rate)
        if stoi_value is None:
            amount_stoi_values_ignored += 1
            print('{} stoi values ignored from a total of {}'.format(amount_stoi_values_ignored, index + 1))
        else:
            stoi_values_per_snr_and_noise_type[snr][noise_type].append(stoi_value)

        pesq_value = pesq(clean_speech, noisy_speech, sample_rate)
        if pesq_value is None:
            amount_pesq_values_ignored += 1
            print('{} pesq values ignored from a total of {}'.format(amount_pesq_values_ignored, index+1))
        else:
            pesq_values_per_snr_and_noise_type[snr][noise_type].append(pesq_value)

    for snr_index, snr in enumerate(snrs_used):
        for noise_index, noise_type in enumerate(noised_used):
            npy_path = join(mixed_path, '..', 'stoi-pesq-values', 'pesq_{}_{}.npy'.format(snr, noise_type))
            with open(npy_path, 'wb') as f:
                pesq_values = np.asarray(pesq_values_per_snr_and_noise_type[snr][noise_type])
                np.save(f, pesq_values)

            npy_path = join(mixed_path, '..', 'stoi-pesq-values', 'stoi_{}_{}.npy'.format(snr, noise_type))
            with open(npy_path, 'wb') as f:
                stoi_values = np.asarray(stoi_values_per_snr_and_noise_type[snr][noise_type])
                np.save(f, stoi_values)

    print('{} stoi values ignored from a total of {}'.format(amount_stoi_values_ignored, amount_audios))
    print('{} pesq values ignored from a total of {}'.format(amount_pesq_values_ignored, amount_audios))


@entry_point.command()
@click.option('--audios-input-dir', default='', required=False, type=str)
def plot_pesq_stoi_histogram(audios_input_dir):
    if not audios_input_dir:
        mixed_path = join(os.getcwd(), '..', 'dataset', 'audios')
    else:
        mixed_path = join(os.getcwd(), '..', 'dataset', audios_input_dir)

    snrs_used = ['-5', '0', '5', '10', '15', '20', 'inf']
    noised_used = ['babble', 'bark', 'meow', 'traffic', 'typing']
    pesq_values_per_snr_and_noise_type = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }
    pesq_values_per_snr = {
        snr: [] for snr in snrs_used
    }

    stoi_values_per_snr_and_noise_type = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }
    stoi_values_per_snr = {
        snr: [] for snr in snrs_used
    }
    for snr_index, snr in enumerate(snrs_used):
        for noise_index, noise_type in enumerate(noised_used):
            npy_path = join(mixed_path, '..', 'stoi-pesq-values', 'pesq_{}_{}.npy'.format(snr, noise_type))
            with open(npy_path, 'rb') as f:
                pesq_values_per_snr_and_noise_type[snr][noise_type] = np.load(f)

            pesq_values_per_snr[snr] = np.concatenate(
                (pesq_values_per_snr[snr], pesq_values_per_snr_and_noise_type[snr][noise_type]), axis=0
            )

            npy_path = join(mixed_path, '..', 'stoi-pesq-values', 'stoi_{}_{}.npy'.format(snr, noise_type))
            with open(npy_path, 'rb') as f:
                stoi_values_per_snr_and_noise_type[snr][noise_type] = np.load(f)

            stoi_values_per_snr[snr] = np.concatenate(
                (stoi_values_per_snr[snr], stoi_values_per_snr_and_noise_type[snr][noise_type]), axis=0
            )

    # Pesq plot by snr and noise type
    pesq_fig, pesq_axs = plt.subplots(len(noised_used), len(snrs_used), figsize=(40, 18), dpi=100, tight_layout=True)
    pesq_fig.suptitle('PESQ Histograma')
    for snr_index, snr in enumerate(snrs_used):
        for noise_index, noise_type in enumerate(noised_used):
            bins_pesq = np.arange(0, 5, 0.25).tolist() + [5]
            pesq_values = np.asarray(pesq_values_per_snr_and_noise_type[snr][noise_type])
            pesq_histogram, _ = np.histogram(pesq_values, bins_pesq, density=False)
            pesq_axs[noise_index, snr_index].bar(bins_pesq[:-1], pesq_histogram, width=0.125, align='center')
            pesq_axs[noise_index, snr_index].title.set_text('SNR: {}, Noise: {}'.format(snr, noise_type))
            pesq_axs[noise_index, snr_index].grid()

    plot_path = join(mixed_path, '..', 'histograms', 'pesq.png')
    pesq_fig.savefig(plot_path)
    plt.close(pesq_fig)

    # Pesq plot by snr
    snr_index_to_plot_index = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 0), 5: (1, 1), 6: (1, 2)}
    pesq_fig, pesq_axs = plt.subplots(2, 4, figsize=(20, 9), dpi=100, tight_layout=True)
    pesq_fig.suptitle('PESQ Histograma')
    for snr_index, snr in enumerate(snrs_used):
        bins_pesq = np.arange(0, 5, 0.25).tolist() + [5]
        pesq_values = np.asarray(pesq_values_per_snr[snr])
        pesq_histogram, _ = np.histogram(pesq_values, bins_pesq, density=False)
        pesq_axs[snr_index_to_plot_index[snr_index]].bar(bins_pesq[:-1], pesq_histogram, width=0.125, align='center')
        pesq_axs[snr_index_to_plot_index[snr_index]].title.set_text('SNR: {}'.format(snr))
        pesq_axs[snr_index_to_plot_index[snr_index]].grid()

    pesq_axs[1, 3].axis('off')
    plot_path = join(mixed_path, '..', 'histograms', 'pesq_aggregated.png')
    pesq_fig.savefig(plot_path)
    plt.close(pesq_fig)

    # Stoi plot by snr and noise type
    stoi_fig, stoi_axs = plt.subplots(len(noised_used), len(snrs_used), figsize=(40, 18), dpi=100, tight_layout=True)
    stoi_fig.suptitle('STOI Histograma')

    for snr_index, snr in enumerate(snrs_used):
        for noise_index, noise_type in enumerate(noised_used):
            bins_stoi = np.arange(0, 1.1, 0.05).tolist() + [1.1]
            stoi_values = np.asarray(stoi_values_per_snr_and_noise_type[snr][noise_type])
            stoi_histogram, _ = np.histogram(stoi_values, bins_stoi, density=False)
            stoi_axs[noise_index, snr_index].bar(bins_stoi[:-1], stoi_histogram, width=0.025, align='center')
            stoi_axs[noise_index, snr_index].title.set_text('SNR: {}, Noise: {}'.format(snr, noise_type))
            stoi_axs[noise_index, snr_index].grid()

    plot_path = join(mixed_path, '..', 'histograms', 'stoi.png')
    stoi_fig.savefig(plot_path)
    plt.close(stoi_fig)

    # Stoi plot by snr
    snr_index_to_plot_index = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 0), 5: (1, 1), 6: (1, 2)}
    stoi_fig, stoi_axs = plt.subplots(2, 4, figsize=(20, 9), dpi=100, tight_layout=True)
    stoi_fig.suptitle('STOI Histograma')
    for snr_index, snr in enumerate(snrs_used):
        bins_stoi = np.arange(0, 1.1, 0.05).tolist() + [1.1]
        stoi_values = np.asarray(stoi_values_per_snr[snr])
        stoi_histogram, _ = np.histogram(stoi_values, bins_stoi, density=False)
        stoi_axs[snr_index_to_plot_index[snr_index]].bar(bins_stoi[:-1], stoi_histogram, width=0.025, align='center')
        stoi_axs[snr_index_to_plot_index[snr_index]].title.set_text('SNR: {}'.format(snr))
        stoi_axs[snr_index_to_plot_index[snr_index]].grid()

    stoi_axs[1, 3].axis('off')
    plot_path = join(mixed_path, '..', 'histograms', 'stoi_aggregated.png')
    stoi_fig.savefig(plot_path)
    plt.close(stoi_fig)


@entry_point.command()
def statistics():
    dataset_path = join(os.getcwd(), '..', 'raw-dataset', 'clean')
    speeches_audio_path = np.array([
        join(dataset_path, file_path) for file_path in os.listdir(dataset_path) if isfile(join(dataset_path, file_path))
    ])

    speeches_duration = []
    for index, speech_audio_path in enumerate(speeches_audio_path):
        try:
            duration = librosa.get_duration(filename=speech_audio_path)
            print('Speech audio number {} read it from a total of {}'.format(index+1, len(speeches_audio_path)))
        except Exception:
            print('Error reading file {}. Continuing with the next audio'.format(speech_audio_path))
            continue

        speeches_duration.append(duration)

    dataset_path = join(os.getcwd(), '..', 'raw-dataset', 'noise')
    noises_audio_path = [file_path for file_path in glob.glob('{}/**/*.wav'.format(dataset_path), recursive=True)]

    noises_duration_per_class = {'babble': [], 'bark': [], 'meow': [], 'traffic': [], 'typing': []}
    for index, noise_audio_path in enumerate(noises_audio_path):
        noise_type = os.path.basename(os.path.dirname(noise_audio_path))
        if noise_type not in noises_duration_per_class:
            continue

        try:
            duration = librosa.get_duration(filename=noise_audio_path)
            print('Noise audio number {} read it from a total of {}'.format(index + 1, len(noise_audio_path)))
        except Exception:
            print('Error reading file {}. Continuing with the next audio'.format(noise_audio_path))
            continue

        noise_type = os.path.basename(os.path.dirname(noise_audio_path))
        noises_duration_per_class[noise_type].append(duration)

    speeches_duration = np.asarray(speeches_duration)
    print('Amount speech audios {}'.format(speeches_duration.shape))
    print('Speeches total duration {}'.format(np.sum(speeches_duration)))
    print('Speeches mean duration {}'.format(np.mean(speeches_duration)))

    for class_type, noises_duration in noises_duration_per_class.items():
        noises_duration = np.asarray(noises_duration)
        print('Amount {} noises audios {}'.format(class_type, noises_duration.shape))
        print('{} noises total duration {}'.format(class_type.capitalize(), np.sum(noises_duration)))
        print('{} noises mean duration {}'.format(class_type.capitalize(), np.mean(noises_duration)))


@entry_point.command()
def noises_psd():
    dataset_path = join(os.getcwd(), '..', 'raw-dataset', 'noise')
    noises = np.asarray(
        [file_path for file_path in glob.glob('{}/**/*.wav'.format(dataset_path), recursive=True)]
    )

    mixed_path = join(os.getcwd(), '..', 'dataset', 'audios')

    amount_noises = len(noises)
    fs = 16000
    noises_type = ['babble', 'bark', 'meow', 'traffic', 'typing']
    noise_type_to_label = {
        'babble': 'Ruido de fondo', 'bark': 'Ladrido', 'meow': 'Maullido', 'traffic': 'Trafico', 'typing': 'Tipeo'
    }

    must_compute_psd = False
    for noise_type in noises_type:
        npy_path = join(mixed_path, '..', 'psd', 'psd_{}.npy'.format(noise_type))
        if not os.path.exists(npy_path):
            must_compute_psd = True

    noise_type_psd_per_noise_type = {noise_type: np.zeros((fs,)) for noise_type in noises_type}
    if must_compute_psd:
        noises_psd_per_noise_type = {noise_type: np.zeros((fs, amount_noises)) for noise_type in noises_type}
        for index, noise_path in enumerate(noises[:amount_noises]):
            print('Computing fft of noise number {} from {}'.format(index+1, len(noises)))
            try:
                noise, noise_sr = librosa.load(noise_path, sr=fs)
                noise_type = os.path.basename(os.path.dirname(noise_path))
            except Exception:
                print('Error reading noise {}. Continuing with the next noise'.format(noise_path))
                continue

            noise_fft = fft(noise, fs)
            noises_psd_per_noise_type[noise_type][:, index] = np.abs(noise_fft)**2

        for noise_type in noises_type:
            psd = np.mean(noises_psd_per_noise_type[noise_type], axis=1)
            noise_type_psd_per_noise_type[noise_type] = psd

            npy_path = join(mixed_path, '..', 'psd', 'psd_{}.npy'.format(noise_type))
            with open(npy_path, 'wb') as f:
                np.save(f, psd)
    else:
        for noise_type in noises_type:
            npy_path = join(mixed_path, '..', 'psd', 'psd_{}.npy'.format(noise_type))
            with open(npy_path, 'rb') as f:
                noise_type_psd_per_noise_type[noise_type] = np.load(f)

    freq_axis = fftfreq(fs, 1 / fs)
    for noise_type in noises_type:
        fig, ax = plt.subplots(tight_layout=True)
        psd = noise_type_psd_per_noise_type[noise_type]
        ax.plot(freq_axis, psd)

        ax.title.set_text('PSD Noise: {}'.format(noise_type))
        ax.grid()
        plot_path = join('..', 'dataset', 'psd', 'psd_{}.png'.format(noise_type))
        fig.savefig(plot_path)
        plt.close(fig)

    noise_type_to_plot_index = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1)}
    psd_agg_fig, psd_agg_axs = plt.subplots(2, 3, figsize=(20, 9), dpi=100, tight_layout=True)
    psd_agg_fig.suptitle('PSD Ruidos')
    for index, noise_type in enumerate(noises_type):
        psd = noise_type_psd_per_noise_type[noise_type]
        psd_agg_axs[noise_type_to_plot_index[index]].plot(freq_axis, psd)
        psd_agg_axs[noise_type_to_plot_index[index]].title.set_text(
            'PSD Noise: {}'.format(noise_type_to_label[noise_type])
        )
        psd_agg_axs[noise_type_to_plot_index[index]].grid()

    psd_agg_axs[1, 2].axis('off')
    plot_path = join(mixed_path, '..', 'psd', 'psd_aggregated.png')
    psd_agg_fig.savefig(plot_path)
    plt.close(psd_agg_fig)


if __name__ == '__main__':
    entry_point()
