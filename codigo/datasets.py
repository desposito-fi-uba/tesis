import csv
import glob
import os
import random
import re
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
@click.option('--use-one-noise-type-per-audio', default=False, required=False, type=bool)
@click.option('--output-dir', default='dataset', required=False, type=str)
def generate(max_amount, env, use_one_noise_type_per_audio, output_dir):
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
        sorted([file_path for file_path in glob.glob(f'{dataset_path}/*.wav', recursive=True)])
    )
    if not use_one_noise_type_per_audio:
        np.random.shuffle(noises)

    if max_amount > 0:
        audios = audios[:max_amount]

    dataset_path = join(os.getcwd(), output_dir)
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
    print('Starting mixing {} clean speeches with {} noises'.format(len(audios), len(noises)))
    for index, audio_path in enumerate(audios):
        _, file_name = os.path.split(audio_path)
        file_name_without_extension, extension = os.path.splitext(file_name)
        try:
            clean_speech, _ = librosa.load(audio_path, sr=sr)
        except Exception:
            print('Error reading file {}. Continuing with the next audio'.format(audio_path))
            continue

        amount_speech_samples = len(clean_speech)
        noise_to_mix = np.zeros((amount_speech_samples, ))
        # noise_to_mix = np.ones((amount_speech_samples, )) * np.finfo(float).eps

        snr_choices = [-5, 0, 5, 10, 15, 20, 40]
        snr = snr_choices[np.random.randint(len(snr_choices))]

        amount_noise_generated = silent_samples
        while amount_noise_generated < amount_speech_samples:
            if amount_noise_used == 0:
                if noise_index + 1 > len(noises):
                    np.random.shuffle(noises)
                    print('Run out of noise data after processing {}'.format(index+1))
                    noise_index = 0

                try:
                    noise_path = noises[noise_index]
                    noise_audio_name = os.path.split(noise_path)[1]
                    matches = re.findall('(.*)_.*', noise_audio_name)
                    if matches:
                        new_noise_type = matches[0]
                        if use_one_noise_type_per_audio and noise_type is not None and new_noise_type != noise_type:
                            noise_type = new_noise_type
                            break
                        noise_type = new_noise_type
                    else:
                        raise RuntimeError(f'Noise audio name {noise_audio_name} without noise type')

                    noise, _ = librosa.load(noise_path, sr=sr)
                except Exception:
                    print('Error reading noise {}. Continuing with the next audio'.format(audio_path))
                    noise_index += 1
                    continue

            reminder_noise_to_generate = amount_speech_samples - amount_noise_generated
            curr_noise_length = len(noise) - amount_noise_used
            if curr_noise_length > reminder_noise_to_generate:
                noise_to_mix[amount_noise_generated:amount_noise_generated+reminder_noise_to_generate] = (
                    noise[amount_noise_used:amount_noise_used+reminder_noise_to_generate]
                )
                amount_noise_used += reminder_noise_to_generate
                amount_noise_generated += reminder_noise_to_generate
            else:
                noise_to_mix[amount_noise_generated:amount_noise_generated+curr_noise_length] = (
                    noise[amount_noise_used:]
                )
                amount_noise_used = 0
                noise_index += 1
                amount_noise_generated += curr_noise_length + silent_samples

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

        if not use_one_noise_type_per_audio:
            noise_type = None

        rows.append([
            join('.', f'audios_{env}', noisy_path_with_out_base_path),
            join('.', f'audios_{env}', clean_path_with_out_base_path),
            join('.', f'audios_{env}', noise_path_with_out_base_path),
            snr, noise_type
        ])

    random.shuffle(rows)

    csv_path = join(dataset_path, f'{env}.csv')
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            writer.writerow(row)


@entry_point.command()
@click.option('--input-dir', default='dataset', prompt='Dataset path', type=str)
def compute_pesq_stoi_histogram(input_dir):
    dataset_path = join(os.getcwd(), input_dir)
    rows = []

    csv_path = join(dataset_path, 'test.csv')
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows.extend([row for row in reader])

    csv_path = join(dataset_path, 'train.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            rows.extend([row for row in reader])

    amount_stoi_values_ignored = 0
    amount_pesq_values_ignored = 0
    amount_audios = len(rows)

    snrs_used = ['-5', '0', '5', '10', '15', '20']
    noises_used = ['Typing', 'Babble', 'Neighbor', 'AirConditioner', 'VacuumCleaner', 'CopyMachine', 'Munching']

    pesq_values_per_snr = {snr: [] for snr in snrs_used}
    stoi_values_per_snr = {snr: [] for snr in snrs_used}

    pesq_values_per_noise_type = {noise_type: [] for noise_type in noises_used}
    stoi_values_per_noise_type = {noise_type: [] for noise_type in noises_used}

    for index, row in enumerate(rows):
        noisy_path, clean_path, noise_path, snr, noise_type = tuple(row)

        clean_speech, sample_rate = librosa.load(join(dataset_path, clean_path), sr=None)
        noisy_speech, sample_rate = librosa.load(join(dataset_path, noisy_path), sr=None)
        noise, sample_rate = librosa.load(join(dataset_path, noise_path), sr=None)

        clean_speech, noisy_speech, noise, _ = remove_not_matched_snr_segments(
            clean_speech, noisy_speech, noise, np.zeros(clean_speech.shape), int(snr), sample_rate
        )

        stoi_value = stoi(clean_speech, noisy_speech, sample_rate)
        if stoi_value is None:
            amount_stoi_values_ignored += 1
            print('{} stoi values ignored from a total of {}'.format(amount_stoi_values_ignored, index + 1))
        else:
            if snr in snrs_used:
                stoi_values_per_snr[snr].append(stoi_value)

            if noise_type in noises_used:
                stoi_values_per_noise_type[noise_type].append(stoi_value)

        pesq_value = pesq(clean_speech, noisy_speech, sample_rate)
        if pesq_value is None:
            amount_pesq_values_ignored += 1
            print('{} pesq values ignored from a total of {}'.format(amount_pesq_values_ignored, index+1))
        else:
            if snr in snrs_used:
                pesq_values_per_snr[snr].append(pesq_value)

            if noise_type in noises_used:
                pesq_values_per_noise_type[noise_type].append(pesq_value)

    for snr_index, snr in enumerate(snrs_used):
        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'pesq_{}.npy'.format(snr))
        with open(npy_path, 'wb') as f:
            pesq_values = np.asarray(pesq_values_per_snr[snr])
            np.save(f, pesq_values)

        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'stoi_{}.npy'.format(snr))
        with open(npy_path, 'wb') as f:
            stoi_values = np.asarray(stoi_values_per_snr[snr])
            np.save(f, stoi_values)

    for noise_type_index, noise_type in enumerate(noises_used):
        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'pesq_by_noise_type_{}.npy'.format(noise_type))
        with open(npy_path, 'wb') as f:
            pesq_values = np.asarray(pesq_values_per_noise_type[noise_type])
            np.save(f, pesq_values)

        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'stoi_by_noise_type_{}.npy'.format(noise_type))
        with open(npy_path, 'wb') as f:
            stoi_values = np.asarray(stoi_values_per_noise_type[noise_type])
            np.save(f, stoi_values)

    print('{} stoi values ignored from a total of {}'.format(amount_stoi_values_ignored, amount_audios))
    print('{} pesq values ignored from a total of {}'.format(amount_pesq_values_ignored, amount_audios))


@entry_point.command()
@click.option('--input-dir', default='dataset', prompt='Dataset path', type=str)
def plot_pesq_stoi_histogram(input_dir):
    dataset_path = join(os.getcwd(), input_dir)

    snrs_used = ['-5', '0', '5', '10', '15', '20']
    pesq_values_per_snr = {snr: [] for snr in snrs_used}
    stoi_values_per_snr = {snr: [] for snr in snrs_used}
    for snr_index, snr in enumerate(snrs_used):
        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'pesq_{}.npy'.format(snr))
        with open(npy_path, 'rb') as f:
            pesq_values_per_snr[snr] = np.load(f)

        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'stoi_{}.npy'.format(snr))
        with open(npy_path, 'rb') as f:
            stoi_values_per_snr[snr] = np.load(f)

    noises_used = ['Typing', 'Babble', 'Neighbor', 'AirConditioner', 'VacuumCleaner', 'CopyMachine', 'Munching']
    pesq_values_per_noise_type = {noise_type: [] for noise_type in noises_used}
    stoi_values_per_noise_type = {noise_type: [] for noise_type in noises_used}
    for noise_type_index, noise_type in enumerate(noises_used):
        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'pesq_by_noise_type_{}.npy'.format(noise_type))
        with open(npy_path, 'rb') as f:
            pesq_values_per_noise_type[noise_type] = np.load(f)

        npy_path = join(dataset_path, 'base_stoi_pesq_values', 'stoi_by_noise_type_{}.npy'.format(noise_type))
        with open(npy_path, 'rb') as f:
            stoi_values_per_noise_type[noise_type] = np.load(f)

    # Pesq plot by snr
    snr_index_to_plot_index = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}
    pesq_fig, pesq_axs = plt.subplots(2, 3, figsize=(20, 9), dpi=100, tight_layout=True)
    pesq_fig.suptitle('PESQ Histograma')
    for snr_index, snr in enumerate(snrs_used):
        bins_pesq = np.arange(0, 5, 0.25).tolist() + [5]
        pesq_values = np.asarray(pesq_values_per_snr[snr])
        pesq_histogram, _ = np.histogram(pesq_values, bins_pesq, density=False)
        pesq_axs[snr_index_to_plot_index[snr_index]].bar(bins_pesq[:-1], pesq_histogram, width=0.125, align='center')
        pesq_axs[snr_index_to_plot_index[snr_index]].title.set_text('SNR: {}'.format(snr))
        pesq_axs[snr_index_to_plot_index[snr_index]].grid()

    plot_path = join(dataset_path, 'base_stoi_pesq_values', 'pesq_aggregated.png')
    pesq_fig.savefig(plot_path)
    plt.close(pesq_fig)

    # Stoi plot by snr
    snr_index_to_plot_index = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}
    stoi_fig, stoi_axs = plt.subplots(2, 3, figsize=(20, 9), dpi=100, tight_layout=True)
    stoi_fig.suptitle('STOI Histograma')
    for snr_index, snr in enumerate(snrs_used):
        bins_stoi = np.arange(0, 1.1, 0.05).tolist() + [1.1]
        stoi_values = np.asarray(stoi_values_per_snr[snr])
        stoi_histogram, _ = np.histogram(stoi_values, bins_stoi, density=False)
        stoi_axs[snr_index_to_plot_index[snr_index]].bar(bins_stoi[:-1], stoi_histogram, width=0.025, align='center')
        stoi_axs[snr_index_to_plot_index[snr_index]].title.set_text('SNR: {}'.format(snr))
        stoi_axs[snr_index_to_plot_index[snr_index]].grid()

    plot_path = join(dataset_path, 'base_stoi_pesq_values', 'stoi_aggregated.png')
    stoi_fig.savefig(plot_path)
    plt.close(stoi_fig)

    # Pesq plot by noise type
    noise_type_index_to_plot_index = {
        0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3)
    }
    pesq_fig, pesq_axs = plt.subplots(2, 4, figsize=(20, 9), dpi=100, tight_layout=True)
    pesq_fig.suptitle('PESQ Histograma')
    for noise_type_index, noise_type in enumerate(noises_used):
        bins_pesq = np.arange(0, 5, 0.25).tolist() + [5]
        pesq_values = np.asarray(pesq_values_per_noise_type[noise_type])
        pesq_histogram, _ = np.histogram(pesq_values, bins_pesq, density=False)
        pesq_axs[noise_type_index_to_plot_index[noise_type_index]].bar(bins_pesq[:-1], pesq_histogram, width=0.125, align='center')
        pesq_axs[noise_type_index_to_plot_index[noise_type_index]].title.set_text('Tipo de ruido: {}'.format(noise_type))
        pesq_axs[noise_type_index_to_plot_index[noise_type_index]].grid()

    pesq_axs[1, 3].axis('off')
    plot_path = join(dataset_path, 'base_stoi_pesq_values', 'pesq_by_noise_type_aggregated.png')
    pesq_fig.savefig(plot_path)
    plt.close(pesq_fig)

    # Stoi plot by noise type
    noise_type_index_to_plot_index = {
        0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3)
    }
    stoi_fig, stoi_axs = plt.subplots(2, 4, figsize=(20, 9), dpi=100, tight_layout=True)
    stoi_fig.suptitle('STOI Histograma')
    for noise_type_index, noise_type in enumerate(noises_used):
        bins_stoi = np.arange(0, 1.1, 0.05).tolist() + [1.1]
        stoi_values = np.asarray(stoi_values_per_noise_type[noise_type])
        stoi_histogram, _ = np.histogram(stoi_values, bins_stoi, density=False)
        stoi_axs[noise_type_index_to_plot_index[noise_type_index]].bar(bins_stoi[:-1], stoi_histogram, width=0.025, align='center')
        stoi_axs[noise_type_index_to_plot_index[noise_type_index]].title.set_text('Tipo de ruido: {}'.format(noise_type))
        stoi_axs[noise_type_index_to_plot_index[noise_type_index]].grid()

    stoi_axs[1, 3].axis('off')
    plot_path = join(dataset_path, 'base_stoi_pesq_values', 'stoi_by_noise_type_aggregated.png')
    stoi_fig.savefig(plot_path)
    plt.close(stoi_fig)


@entry_point.command()
@click.option('--env', default='train', required=False, type=str)
def statistics(env):
    dataset_path = join(os.getcwd(), 'dataset')
    rows = []
    speeches_duration = []

    csv_path = join(dataset_path, f'{env}.csv')
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows.extend([row for row in reader])

    for index, row in enumerate(rows):
        _, clean_path, _, _, _ = tuple(row)

        duration = librosa.get_duration(filename=join(dataset_path, clean_path))
        print('Speech audio number {} read it from a total of {}'.format(index+1, len(rows)))

        speeches_duration.append(duration)

    speeches_duration = np.asarray(speeches_duration)
    print('Amount speech audios {}'.format(speeches_duration.shape))
    print('Speeches total duration {}'.format(np.sum(speeches_duration)))
    print('Speeches mean duration {}'.format(np.mean(speeches_duration)))


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
