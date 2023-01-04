import glob
import math
import os
import random

import click
import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import fft, ifft
from scipy.signal import convolve, stft
from torch import sigmoid
from torch.utils.data import DataLoader

import runsettings
from adaptivealgorithms import rls
from adaptivefilterdatasethandler import AdaptiveFilteringDataset
from audiolib import segmental_snr_mixer
from constants import AudioType
from realtimednnfilterdatasethandler import RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures
from utils import pesq, remove_not_matched_snr_segments, stoi


@click.group()
def entry_point():
    pass


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
def af_plot_weights(input_dir):
    dataset_path = input_dir

    constant_filter = False
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    test_dataset = AdaptiveFilteringDataset(
        dataset_path, test_dataset_path, runsettings.fs, runsettings.af_windows_time_size,
        runsettings.overlap_percentage, randomize=False,
        max_samples=runsettings.test_on_n_samples, constant_filter=constant_filter
    )
    test_data_loader = DataLoader(test_dataset, batch_size=runsettings.af_batch_size)
    test_data_iterator = iter(test_data_loader)
    plot_dim = 2, 2
    fig, axs = plt.subplots(*plot_dim, figsize=(20, 9), dpi=100, tight_layout=True)
    plotted = 0
    while plotted != 4:
        sample_batched = next(test_data_iterator)

        sample_idx = sample_batched['sample_idx'].numpy()[0]
        snr = test_dataset.get_audio_data(sample_idx)['snr']
        noise_type = test_dataset.get_audio_data(sample_idx)['noise_type']

        if snr != 20.0:
            continue

        noisy_frame = sample_batched['noisy_speech'].numpy()[0]
        correlated_noise_frame = sample_batched['correlated_noise'].numpy()[0]

        noise_estimation, filtered_speech_frame, weights_n, _, _ = rls(
            correlated_noise_frame, noisy_frame, runsettings.filter_size, lmda=runsettings.forgetting_rate
        )

        audio_name = test_dataset.get_audio_data(sample_idx)['audio_name']
        axs[np.unravel_index(plotted, plot_dim)].plot(np.transpose(weights_n))
        axs[np.unravel_index(plotted, plot_dim)].set_title(
            'Audio {} - SNR {} - Tipo {}'.format(audio_name, snr, noise_type)
        )

        plotted += 1

    fig.savefig('./store/weights.png')
    plt.close(fig)


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
def af_plot_signals(input_dir):
    dataset_path = input_dir

    sample_idx = 0
    split_in_frames = False
    constant_filter = False
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    test_dataset = AdaptiveFilteringDataset(
        dataset_path, test_dataset_path, runsettings.fs, runsettings.af_windows_time_size,
        runsettings.overlap_percentage, randomize=False,
        max_samples=runsettings.test_on_n_samples, constant_filter=constant_filter
    )
    test_data_loader = DataLoader(test_dataset, batch_size=runsettings.af_batch_size)
    test_data_iterator = iter(test_data_loader)

    test_dataset.forward_to_audio_sample_idx(sample_idx)
    sample_batched = next(test_data_iterator)

    noisy_frame = sample_batched['noisy_speech'].numpy()[0]
    clean_speech_frame = sample_batched['clean_speech'].numpy()[0]
    correlated_noise_frame = sample_batched['correlated_noise'].numpy()[0]
    noise_frame = sample_batched['noise'].numpy()[0]
    snr = test_dataset.get_audio_data(sample_idx)['snr']
    noise_type = test_dataset.get_audio_data(sample_idx)['noise_type']
    audio_name = test_dataset.get_audio_data(sample_idx)['audio_name']

    noise_estimation, filtered_speech_frame, weights_n, _, _ = rls(
        correlated_noise_frame, noisy_frame, runsettings.filter_size, lmda=runsettings.forgetting_rate
    )

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 9), dpi=100, tight_layout=True)
    fig.suptitle('Audio {} - SNR {} - Type {}'.format(audio_name, snr, noise_type))
    axs[0].plot(np.transpose(weights_n))
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('Coeficientes del filtro')
    axs[1].plot(noise_frame, label='Ruido')
    axs[1].plot(noise_estimation, label='Estimación del ruido')
    axs[1].legend()
    axs[1].set_title('Señales de ruido')
    axs[1].grid()
    axs[2].plot(clean_speech_frame, label='Señal de habla')
    axs[2].plot(filtered_speech_frame, label='Estimación de la señal de habla')
    axs[2].legend()
    axs[2].grid()
    axs[2].set_title('Señales de habla')

    fig.savefig('./store/signals.png'.format(audio_name, noise_type, snr))
    plt.close(fig)

    x_lim = (19000, 26000)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 9), dpi=100, tight_layout=True)
    fig.suptitle('Audio {} - SNR {} - Type {}'.format(audio_name, snr, noise_type))
    axs[0].plot(np.transpose(weights_n))
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('Coeficientes del filtro')
    axs[0].set_xlim(*x_lim)
    axs[1].plot(noise_estimation, label='Estimación del ruido')
    axs[1].plot(noise_frame, label='Ruido')
    axs[1].legend()
    axs[1].set_title('Señales de ruido')
    axs[1].grid()
    axs[1].set_xlim(*x_lim)
    axs[2].plot(filtered_speech_frame, label='Estimación de la señal de habla')
    axs[2].plot(clean_speech_frame, label='Señal de habla')
    axs[2].legend()
    axs[2].grid()
    axs[2].set_title('Señales de habla')
    axs[2].set_xlim(*x_lim)

    fig.savefig('./store/zoomed_signals.png'.format(audio_name, noise_type, snr))
    plt.close(fig)


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
def plot_af_stationary_noise(input_dir):
    raw_data_set_path = os.path.join('.', '..', 'raw-dataset')
    noises_path = np.asarray(
        [file_path for file_path in glob.glob('{}/noise/typing/*.wav'.format(raw_data_set_path), recursive=True)]
    )
    audio_path = os.path.join(raw_data_set_path, 'clean', 'p234_001.wav')

    clean_speech, _ = librosa.load(audio_path, sr=16000)

    snrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    fig, axs = plt.subplots(2, 3, figsize=(20, 9), dpi=100, tight_layout=True)
    for index, (snr, snr) in enumerate(zip(snrs, snrs)):
        print(f'Start estimating learning curve for SNR {snr}')
        correlation_filter = np.random.random((10,))
        num_experiments = 500
        estimation_errors = np.ones((num_experiments, len(clean_speech))) * 1e-20
        noises = np.ones((num_experiments, len(clean_speech))) * 1e-20
        for j in range(num_experiments):
            if (j+1) % 100 == 0:
                print(f'Start estimating learning curve for SNR {snr}. Experiment number {j+1}')

            noise_path = random.choice(noises_path)
            noise, _ = librosa.load(noise_path, sr=16000)
            start_index = random.randint(0, len(noise) - len(clean_speech))
            end_index = start_index + len(clean_speech)
            noise = noise[start_index:end_index]
            clean, noise, noisy, _ = segmental_snr_mixer(clean_speech, noise, snr)

            correlated_noise = convolve(noise, correlation_filter, mode='same')

            noise_estimation, filtered, weights_n, _, _ = rls(
                correlated_noise, noisy, runsettings.filter_size, lmda=runsettings.forgetting_rate
            )
            estimation_error = np.abs(filtered - clean) ** 2
            estimation_error_smoothed = exponential_moving_average_smoothing(estimation_error, 0.9995)
            estimation_errors[j, :] = estimation_error_smoothed

            noise_abs = np.abs(noise) ** 2
            noise_abs_smoothed = exponential_moving_average_smoothing(noise_abs, 0.995)
            noises[j, :] = noise_abs_smoothed

        ensamble_learning_curve = np.mean(estimation_errors, axis=0)
        noises_mean = np.mean(noises, axis=0)

        axs[np.unravel_index(index, (2, 3), order='F')].plot(ensamble_learning_curve, label=f'ECM')
        axs[np.unravel_index(index, (2, 3), order='F')].plot(noises_mean, label=f'Nivel de ruido')
        axs[np.unravel_index(index, (2, 3), order='F')].legend()
        axs[np.unravel_index(index, (2, 3), order='F')].grid()
        axs[np.unravel_index(index, (2, 3), order='F')].set_title(f'SNR {snr}')
        print(f'End estimating learning curve for SNR {snr}')

    fig.suptitle('ECM y Nivel de ruido')
    fig.savefig('./store/curva_de_aprendizaje.png')
    plt.close(fig)


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
def plot_af_frequency_response(input_dir):
    raw_data_set_path = os.path.join('.', '..', 'raw-dataset')
    noises_path = np.asarray(
        [file_path for file_path in glob.glob('{}/noise/typing/*.wav'.format(raw_data_set_path), recursive=True)]
    )
    audio_path = os.path.join(raw_data_set_path, 'clean', 'p234_001.wav')

    clean_speech, _ = librosa.load(audio_path, sr=16000)

    noise_path = random.choice(noises_path)
    noise, _ = librosa.load(noise_path, sr=16000)
    start_index = random.randint(0, len(noise) - len(clean_speech))
    end_index = start_index + len(clean_speech)
    noise = noise[start_index:end_index]

    correlation_filter = np.random.random((10,))

    # -5 dB
    clean_minus_five, noise_minus_five, noisy_minus_five, _ = segmental_snr_mixer(clean_speech, noise, -5)
    correlated_noise_minus_five = convolve(noise_minus_five, correlation_filter, mode='same')

    noise_estimation_minus_five, filtered_minus_five, _, _, _ = rls(
        correlated_noise_minus_five, noisy_minus_five, runsettings.filter_size, lmda=runsettings.forgetting_rate
    )

    # 20 dB
    clean_twenty, noise_twenty, noisy_twenty, _ = segmental_snr_mixer(clean_speech, noise, 20)
    correlated_noise_twenty = convolve(noise_twenty, correlation_filter, mode='same')

    noise_estimation_twenty, filtered_twenty, _, _, _ = rls(
        correlated_noise_twenty, noisy_twenty, runsettings.filter_size, lmda=runsettings.forgetting_rate
    )

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 9), dpi=100, tight_layout=True)
    axs[0, 0].plot(np.abs(fft(clean_minus_five, 2**12))[:100])
    axs[0, 0].set_title('Clean -5 dB')
    axs[0, 1].plot(np.abs(fft(filtered_minus_five, 2 ** 12))[:100])
    axs[0, 1].set_title('Filtered -5 dB')
    axs[1, 0].plot(np.abs(fft(clean_twenty, 2 ** 12))[:100])
    axs[1, 0].set_title('Clean 20 dB')
    axs[1, 1].plot(np.abs(fft(filtered_twenty, 2 ** 12))[:100])
    axs[1, 1].set_title('Filtered 20 dB')

    fig.savefig('./store/frequency_response.png')
    plt.close(fig)

    print(f'PESQ Clean Noisy -5 dB {pesq(clean_minus_five, noisy_minus_five, 16000)}')
    print(f'PESQ Clean Filtered -5 dB {pesq(clean_minus_five, filtered_minus_five, 16000)}')

    print(f'PESQ Clean Noisy 20 dB {pesq(clean_twenty, noisy_twenty, 16000)}')
    print(f'PESQ Clean Filtered 20 dB {pesq(clean_twenty, filtered_twenty, 16000)}')


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
@click.option('--model', required=False, help='Name of the model to be used to predict', type=str)
def dnn_plot_example_stft_features(input_dir, model):
    dataset_path = input_dir

    test_dataset_path = os.path.join(dataset_path, 'test.csv')

    model_path = os.path.join(os.getcwd(), 'trained-models', model)
    runsettings.net.load_state_dict(torch.load(model_path))

    test_dataset = RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(
        dataset_path, test_dataset_path, runsettings.fs, runsettings.windows_time_size,
        runsettings.overlap_percentage, runsettings.fft_points, runsettings.time_feature_size,
        randomize=runsettings.test_randomize_data, max_samples=runsettings.test_on_n_samples,
        normalize=runsettings.normalize_data, predict_on_time_windows=runsettings.predict_on_time_windows
    )

    test_data_loader = DataLoader(test_dataset, batch_size=runsettings.test_batch_size)
    test_data_iterator = iter(test_data_loader)

    runsettings.net.to(runsettings.device)
    runsettings.net.eval()

    with torch.no_grad():
        while True:

            sample_batched = next(test_data_iterator)
            noisy_speeches = sample_batched['noisy_speech']
            samples_idx = sample_batched['sample_idx']

            noisy_speeches = noisy_speeches.to(runsettings.device)

            outputs, _ = runsettings.net(noisy_speeches)
            if runsettings.normalize_data:
                outputs = sigmoid(outputs)

            accumulated_samples_idx = test_dataset.accumulate_filtered_frames(outputs.cpu(), samples_idx)
            if accumulated_samples_idx:
                sample_idx = accumulated_samples_idx[0]
                break

    _, _, noisy_speech, _, filtered_speech, _, _ = test_dataset.get_sample_data(sample_idx)
    freq_time_shape = noisy_speech[:, :, 0].shape
    frequency_axis, time_axis = test_dataset.get_frq_time_axis(freq_time_shape[0], freq_time_shape[1])
    plot_dim = 4, 4
    total_plots = 16
    fig, axs = plt.subplots(*plot_dim, figsize=(20, 9), dpi=100, tight_layout=True)
    for plot_index, frame_index in zip(range(total_plots), range(16, 16 + total_plots)):
        axs[np.unravel_index(plot_index, plot_dim)].pcolormesh(
            time_axis, frequency_axis, noisy_speech[:, :, frame_index], shading='gouraud'
        )
        axs[np.unravel_index(plot_index, plot_dim)].axis('off')

    fig.savefig('./store/example_stft_feature.png')
    plt.close(fig)

    filtered_audio = test_dataset.get_audio(sample_idx, AudioType.FILTERED)
    filtered_magnitude_stft = test_dataset.get_stft(filtered_audio)

    fig, axs = plt.subplots(figsize=(20, 9), dpi=100, tight_layout=True)
    freq_time_shape = filtered_magnitude_stft.shape
    frequency_axis, time_axis = test_dataset.get_frq_time_axis(freq_time_shape[0], freq_time_shape[1])
    axs.pcolormesh(
        time_axis, frequency_axis, filtered_magnitude_stft, shading='gouraud'
    )
    axs.axis('off')

    fig.savefig('./store/example_output_stft.png')
    plt.close(fig)


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
def plot_dnn_stationary_noise(input_dir):
    dataset_path = input_dir
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    snrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    ecm_by_snr = {snr: None for snr in snrs}
    noise_level_by_snr = {snr: None for snr in snrs}
    for snr_index, snr in enumerate(snrs):
        print(f'Estimating learning curve for SNR {snr}')
        estimation_errors = []
        noises = []
        amount_used = 0
        samples = pd.read_csv(test_dataset_path)
        num_samples = len(samples)
        for sample_idx in range(num_samples):
            curr_snr = samples.iloc[sample_idx, 3]

            if math.isinf(snr) or curr_snr != snr:
                continue

            clean_path = os.path.join(dataset_path, samples.iloc[sample_idx, 1])
            audio_name = clean_path.replace('-clean.wav', '.wav').replace(dataset_path, '')[1:]
            filtered_path = os.path.join(dataset_path, '..', 'filtered', 'dnn', audio_name)

            noise_path = os.path.join(dataset_path, samples.iloc[sample_idx, 2])
            noisy_path = os.path.join(dataset_path, samples.iloc[sample_idx, 0])

            filtered, _ = librosa.load(filtered_path, sr=None)
            clean, _ = librosa.load(clean_path, sr=None)
            noise, _ = librosa.load(noise_path, sr=None)
            noisy, _ = librosa.load(noisy_path, sr=None)

            clean, noisy, noise, filtered = remove_not_matched_snr_segments(
                clean, noisy, noise, filtered, int(snr), runsettings.fs
            )
            if clean.size == 0:
                continue

            estimation_error = np.mean(np.abs(filtered - clean) ** 2)
            estimation_errors.append(estimation_error)

            noise = np.mean(np.abs(noise) ** 2)
            noises.append(noise)

            amount_used += 1

        print(f'End estimating learning curve for SNR {snr} with {len(estimation_errors)} samples')
        estimation_errors = np.stack(estimation_errors, axis=0)
        estimation_errors = np.mean(estimation_errors, axis=0)
        noises = np.stack(noises, axis=0)
        noises = np.mean(noises, axis=0)
        noise_level_by_snr[snr] = noises
        ecm_by_snr[snr] = estimation_errors
        print(f'End estimating learning curve for SNR {snr}')

    fig, axs = plt.subplots(figsize=(20, 9), dpi=100, tight_layout=True)
    axs.plot(snrs, ecm_by_snr.values(), label='ECM', marker='o', linestyle='dashed')
    axs.plot(snrs, noise_level_by_snr.values(), label='Nivel de ruido', marker='o', linestyle='dashed')
    axs.set_title('ECM y Nivel de Rudio')
    axs.legend()
    axs.grid()
    fig.savefig('./store/dnn_ecm_and_noise_level.png')
    plt.close(fig)


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
def plot_pesq_stoi(input_dir):
    dataset_path = input_dir
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    samples = pd.read_csv(test_dataset_path)
    num_samples = len(samples)

    snrs_used = ['-5', '0', '5', '10', '15', '20', 'inf']

    noisy_pesq = {snr: [] for snr in snrs_used}
    noisy_stoi = {snr: [] for snr in snrs_used}

    filtered_pesq = {snr: [] for snr in snrs_used}
    filtered_stoi = {snr: [] for snr in snrs_used}

    noisy_pesq_total = []
    noisy_stoi_total = []

    filtered_pesq_total = []
    filtered_stoi_total = []

    def plot_result(audio_num):
        for snr in snrs_used:
            if noisy_pesq[snr]:
                print(f'[{audio_num}] Noisy PESQ para SNR {snr}: {np.asarray(noisy_pesq[snr]).mean()}')
            else:
                print(f'[{audio_num}] Noisy PESQ para SNR {snr}: None')

            if filtered_pesq[snr]:
                print(f'[{audio_num}] Filtered PESQ para SNR {snr}: {np.asarray(filtered_pesq[snr]).mean()}')
            else:
                print(f'[{audio_num}] Filtered PESQ para SNR {snr}: None')

            print('\n')

            if noisy_stoi[snr]:
                print(f'[{audio_num}] Noisy STOI para SNR {snr}: {np.asarray(noisy_stoi[snr]).mean()}')
            else:
                print(f'[{audio_num}] Noisy STOI para SNR {snr}: None')

            if filtered_stoi[snr]:
                print(f'[{audio_num}] Filtered STOI para SNR {snr}: {np.asarray(filtered_stoi[snr]).mean()}')
            else:
                print(f'[{audio_num}] Filtered STOI para SNR {snr}: None')

            print('\n\n')

        if noisy_pesq_total:
            print(f'[{audio_num}] Noisy PESQ: {np.asarray(noisy_pesq_total).mean()}')
        else:
            print(f'[{audio_num}] Noisy PESQ: None')

        if filtered_pesq_total:
            print(f'[{audio_num}] Filtered PESQ: {np.asarray(filtered_pesq_total).mean()}')
        else:
            print(f'[{audio_num}] Filtered PESQ: None')

        print('\n')

        if noisy_stoi_total:
            print(f'[{audio_num}] Noisy STOI: {np.asarray(noisy_stoi_total).mean()}')
        else:
            print(f'[{audio_num}] Noisy STOI: None')

        if filtered_stoi_total:
            print(f'[{audio_num}] Filtered STOI: {np.asarray(filtered_stoi_total).mean()}')
        else:
            print(f'[{audio_num}] Filtered STOI: None')

        print('\n\n')

    for sample_idx in range(num_samples):
        snr = str(int(samples.iloc[sample_idx, 3])) if not math.isinf(samples.iloc[sample_idx, 3]) else 'inf'

        clean_path = os.path.join(dataset_path, samples.iloc[sample_idx, 1])
        clean, sr = librosa.load(clean_path, sr=None)

        audio_name = clean_path.replace('-clean.wav', '.wav').replace(dataset_path, '')[1:]

        filtered_path = os.path.join(dataset_path, '..', 'filtered', 'dnn', audio_name)
        filtered, _ = librosa.load(filtered_path, sr=None)

        noise = None
        if snr != 'inf':
            noise_path = os.path.join(dataset_path, samples.iloc[sample_idx, 2])
            noise, _ = librosa.load(noise_path, sr=None)

        noisy_path = os.path.join(dataset_path, samples.iloc[sample_idx, 0])
        noisy, _ = librosa.load(noisy_path, sr=None)

        if noise is not None:
            clean, noisy, noise, filtered = remove_not_matched_snr_segments(
                clean, noisy, noise, filtered, int(snr), runsettings.fs
            )

        pesq_value = pesq(clean, noisy, sr)
        if pesq_value:
            noisy_pesq[snr].append(pesq_value)
            noisy_pesq_total.append(pesq_value)

        stoi_value = stoi(clean, noisy, sr)
        if stoi_value:
            noisy_stoi[snr].append(stoi_value)
            noisy_stoi_total.append(stoi_value)

        pesq_value = pesq(clean, filtered, sr)
        if pesq_value:
            filtered_pesq[snr].append(pesq_value)
            filtered_pesq_total.append(pesq_value)

        stoi_value = stoi(clean, filtered, sr)
        if stoi_value:
            filtered_stoi[snr].append(stoi_value)
            filtered_stoi_total.append(stoi_value)

        if (sample_idx + 1) % 100 == 0:
            plot_result(sample_idx + 1)

    plot_result(num_samples)


def exponential_moving_average_smoothing(x, weight):
    last = x[0]
    smoothed = list()
    for point in x:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def convolution_smoothing(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


stoi_threshold = {
    '-5': (0.2, 0.7),
    '0': (0.2, 0.75),
    '5': (0.2, 0.8),
    '10': (0.2, 0.85),
    '15': (0.2, 0.9),
    '20': (0.2, 0.97),
    'inf': (0.2, 1)
}


if __name__ == '__main__':
    entry_point()
