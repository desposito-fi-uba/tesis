import math
import os

import click
import librosa
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from adaptivealgorithms import rls
from adaptivefilterdatasethandler import AdaptiveFilteringDataset
from runsettings import RunSettings
from utils import pesq, stoi, remove_not_matched_snr_segments


@click.group()
def entry_point():
    pass


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
def af_plot_weights(input_dir):
    run_settings = RunSettings()
    dataset_path = input_dir

    constant_filter = False
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    test_dataset = AdaptiveFilteringDataset(
        dataset_path, test_dataset_path, run_settings.fs, run_settings.af_windows_time_size,
        run_settings.overlap_percentage, randomize=False,
        max_samples=run_settings.test_on_n_samples, constant_filter=constant_filter
    )
    test_data_loader = DataLoader(test_dataset, batch_size=run_settings.af_batch_size)
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
            correlated_noise_frame, noisy_frame, run_settings.filter_size, lmda=run_settings.forgetting_rate
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
@click.option('--filter-type', prompt='Filter type: dnn or af', default='dnn', type=str)
def plot_ecm_and_noise_level(input_dir, filter_type):
    run_settings = RunSettings()
    dataset_path = input_dir
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    snrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    ecm_by_snr = {snr: None for snr in snrs}
    noise_level_by_snr = {snr: None for snr in snrs}
    for snr_index, snr in enumerate(snrs):
        print(f'Estimating for SNR {snr}')
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
            audio_name = os.path.split(clean_path)[1].replace('-clean.wav', '.wav')
            filtered_path = os.path.join(dataset_path, 'filtered', filter_type, audio_name)

            if not os.path.exists(filtered_path):
                continue

            noise_path = os.path.join(dataset_path, samples.iloc[sample_idx, 2])
            noisy_path = os.path.join(dataset_path, samples.iloc[sample_idx, 0])

            filtered, _ = librosa.load(filtered_path, sr=None)
            clean, _ = librosa.load(clean_path, sr=None)
            noise, _ = librosa.load(noise_path, sr=None)
            noisy, _ = librosa.load(noisy_path, sr=None)

            clean, noisy, noise, filtered = remove_not_matched_snr_segments(
                clean, noisy, noise, filtered, int(snr), RunSettings().fs
            )
            if clean.size == 0:
                print(
                    f'Ignoring sample {sample_idx} because of empty length after removal of not matched snr segments'
                )
                continue

            estimation_error = np.mean(np.abs(filtered - clean) ** 2)
            estimation_errors.append(estimation_error)

            noise = np.mean(np.abs(noise) ** 2)
            noises.append(noise)

            amount_used += 1

        print(f'End estimating for SNR {snr} with {len(estimation_errors)} samples')
        estimation_errors = np.stack(estimation_errors, axis=0)
        estimation_errors = np.mean(estimation_errors, axis=0)
        noises = np.stack(noises, axis=0)
        noises = np.mean(noises, axis=0)
        noise_level_by_snr[snr] = noises
        ecm_by_snr[snr] = estimation_errors
        print(f'End estimating for SNR {snr}')

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.plot(snrs, ecm_by_snr.values(), label='ECM', marker='o', linestyle='dashed')
    axs.plot(snrs, noise_level_by_snr.values(), label='Nivel de ruido', marker='o', linestyle='dashed')
    axs.set_yticklabels(['{:.1e}'.format(float(x)) for x in axs.get_yticks().tolist()])
    axs.set_xlabel('SNR [dB]')
    axs.set_title('ECM y Nivel de Rudio')
    axs.legend()
    axs.grid()
    fig.savefig(f'./store/{filter_type}_ecm_and_noise_level.png')
    plt.close(fig)


@entry_point.command()
@click.option('--input-dir', prompt='Dataset path', type=str)
@click.option('--filter-type', prompt='Filter type: dnn or af', default='dnn', type=str)
def plot_pesq_stoi(input_dir, filter_type):
    dataset_path = input_dir
    dataset_dir_name = os.path.split(dataset_path)[1]
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    samples = pd.read_csv(test_dataset_path)
    num_samples = len(samples)

    snrs_used = ['-5', '0', '5', '10', '15', '20']
    noises_used = ['Typing', 'Babble', 'Neighbor', 'AirConditioner', 'VacuumCleaner', 'CopyMachine', 'Munching']

    exclude_snr = []
    must_remove_not_matched_snr_segments = False
    if filter_type == 'dnn':
        exclude_snr = [] # ['-5', '0']
        must_remove_not_matched_snr_segments = True
    elif filter_type == 'af':
        exclude_snr = [] # ['-5']
        must_remove_not_matched_snr_segments = True

    noisy_pesq = {snr: [] for snr in snrs_used}
    noisy_stoi = {snr: [] for snr in snrs_used}

    filtered_pesq = {snr: [] for snr in snrs_used}
    filtered_stoi = {snr: [] for snr in snrs_used}

    noisy_pesq_by_noise_type = {noise_type: [] for noise_type in noises_used}
    noisy_stoi_by_noise_type = {noise_type: [] for noise_type in noises_used}

    filtered_pesq_by_noise_type = {noise_type: [] for noise_type in noises_used}
    filtered_stoi_by_noise_type = {noise_type: [] for noise_type in noises_used}

    noisy_pesq_total = []
    noisy_stoi_total = []

    filtered_pesq_total = []
    filtered_stoi_total = []

    def print_result(audio_num):
        with open(f'./store/{filter_type}_{dataset_dir_name}.log', 'w') as f:

            for snr in snrs_used:
                if noisy_pesq[snr]:
                    print(f'[{audio_num}] Noisy PESQ para SNR {snr}: {np.asarray(noisy_pesq[snr]).mean()}')
                    print(f'[{audio_num}] Noisy PESQ para SNR {snr}: {np.asarray(noisy_pesq[snr]).mean()}', file=f)
                else:
                    print(f'[{audio_num}] Noisy PESQ para SNR {snr}: None')
                    print(f'[{audio_num}] Noisy PESQ para SNR {snr}: None', file=f)

                if filtered_pesq[snr]:
                    print(f'[{audio_num}] Filtered PESQ para SNR {snr}: {np.asarray(filtered_pesq[snr]).mean()}')
                    print(f'[{audio_num}] Filtered PESQ para SNR {snr}: {np.asarray(filtered_pesq[snr]).mean()}', file=f)
                else:
                    print(f'[{audio_num}] Filtered PESQ para SNR {snr}: None')
                    print(f'[{audio_num}] Filtered PESQ para SNR {snr}: None', file=f)

                print('\n')
                print('\n', file=f)

                if noisy_stoi[snr]:
                    print(f'[{audio_num}] Noisy STOI para SNR {snr}: {np.asarray(noisy_stoi[snr]).mean()}')
                    print(f'[{audio_num}] Noisy STOI para SNR {snr}: {np.asarray(noisy_stoi[snr]).mean()}', file=f)
                else:
                    print(f'[{audio_num}] Noisy STOI para SNR {snr}: None')
                    print(f'[{audio_num}] Noisy STOI para SNR {snr}: None', file=f)

                if filtered_stoi[snr]:
                    print(f'[{audio_num}] Filtered STOI para SNR {snr}: {np.asarray(filtered_stoi[snr]).mean()}')
                    print(f'[{audio_num}] Filtered STOI para SNR {snr}: {np.asarray(filtered_stoi[snr]).mean()}', file=f)
                else:
                    print(f'[{audio_num}] Filtered STOI para SNR {snr}: None')
                    print(f'[{audio_num}] Filtered STOI para SNR {snr}: None', file=f)

                print('\n\n')
                print('\n\n', file=f)

            if noisy_pesq_total:
                print(f'[{audio_num}] Noisy PESQ: {np.asarray(noisy_pesq_total).mean()}')
                print(f'[{audio_num}] Noisy PESQ: {np.asarray(noisy_pesq_total).mean()}', file=f)
            else:
                print(f'[{audio_num}] Noisy PESQ: None')
                print(f'[{audio_num}] Noisy PESQ: None', file=f)

            if filtered_pesq_total:
                print(f'[{audio_num}] Filtered PESQ: {np.asarray(filtered_pesq_total).mean()}')
                print(f'[{audio_num}] Filtered PESQ: {np.asarray(filtered_pesq_total).mean()}', file=f)
            else:
                print(f'[{audio_num}] Filtered PESQ: None')
                print(f'[{audio_num}] Filtered PESQ: None', file=f)

            print('\n')
            print('\n', file=f)

            if noisy_stoi_total:
                print(f'[{audio_num}] Noisy STOI: {np.asarray(noisy_stoi_total).mean()}')
                print(f'[{audio_num}] Noisy STOI: {np.asarray(noisy_stoi_total).mean()}', file=f)
            else:
                print(f'[{audio_num}] Noisy STOI: None')
                print(f'[{audio_num}] Noisy STOI: None', file=f)

            if filtered_stoi_total:
                print(f'[{audio_num}] Filtered STOI: {np.asarray(filtered_stoi_total).mean()}')
                print(f'[{audio_num}] Filtered STOI: {np.asarray(filtered_stoi_total).mean()}', file=f)
            else:
                print(f'[{audio_num}] Filtered STOI: None')
                print(f'[{audio_num}] Filtered STOI: None', file=f)

            print('\n\n')
            print('\n\n', file=f)

    for sample_idx in range(num_samples):
        noise_type = samples.iloc[sample_idx, 4] if samples.iloc[sample_idx, 4] else 'Undefined'

        snr = str(int(samples.iloc[sample_idx, 3])) if not math.isinf(samples.iloc[sample_idx, 3]) else 'inf'
        if snr not in snrs_used:
            continue

        clean_path = os.path.join(dataset_path, samples.iloc[sample_idx, 1])
        clean, sr = librosa.load(clean_path, sr=None)

        audio_name = clean_path.replace('-clean.wav', '.wav').replace(dataset_path, '')[1:]
        audio_name = os.path.basename(audio_name)

        filtered_path = os.path.join(dataset_path, 'filtered', filter_type, audio_name)
        if not os.path.exists(filtered_path):
            continue

        filtered, _ = librosa.load(filtered_path, sr=None)

        noise = None
        if snr != 'inf':
            noise_path = os.path.join(dataset_path, samples.iloc[sample_idx, 2])
            noise, _ = librosa.load(noise_path, sr=None)

        noisy_path = os.path.join(dataset_path, samples.iloc[sample_idx, 0])
        noisy, _ = librosa.load(noisy_path, sr=None)

        if must_remove_not_matched_snr_segments:
            clean, noisy, noise, filtered = remove_not_matched_snr_segments(
                clean, noisy, noise, filtered, int(snr), RunSettings().fs
            )
            if clean.size == 0:
                print(
                    f'Ignoring sample {sample_idx} because of empty length after removal of not matched snr segments'
                )
                continue

        pesq_value = pesq(clean, noisy, sr)
        if pesq_value:
            noisy_pesq[snr].append(pesq_value)
            noisy_pesq_total.append(pesq_value)

            if noise_type in noises_used and snr not in exclude_snr:
                noisy_pesq_by_noise_type[noise_type].append(pesq_value)

        stoi_value = stoi(clean, noisy, sr)
        if stoi_value:
            noisy_stoi[snr].append(stoi_value)
            noisy_stoi_total.append(stoi_value)

            if noise_type in noises_used and snr not in exclude_snr:
                noisy_stoi_by_noise_type[noise_type].append(stoi_value)

        pesq_value = pesq(clean, filtered, sr)
        if pesq_value:
            filtered_pesq[snr].append(pesq_value)
            filtered_pesq_total.append(pesq_value)

            if noise_type in noises_used and snr not in exclude_snr:
                filtered_pesq_by_noise_type[noise_type].append(pesq_value)

        stoi_value = stoi(clean, filtered, sr)
        if stoi_value:
            filtered_stoi[snr].append(stoi_value)
            filtered_stoi_total.append(stoi_value)

            if noise_type in noises_used and snr not in exclude_snr:
                filtered_stoi_by_noise_type[noise_type].append(stoi_value)

        if (sample_idx + 1) % 100 == 0:
            print_result(sample_idx + 1)

    print_result(num_samples)

    mean_noisy_pesq = [np.asarray(noisy_pesq[snr]).mean() for snr in snrs_used]
    mean_filtered_pesq = [np.asarray(filtered_pesq[snr]).mean() for snr in snrs_used]

    with open(f'./store/{filter_type}_{dataset_dir_name}_noisy_pesq_by_snr.npy', 'wb') as f:
        np.save(f, mean_noisy_pesq)

    with open(f'./store/{filter_type}_{dataset_dir_name}_filtered_pesq_by_snr.npy', 'wb') as f:
        np.save(f, mean_filtered_pesq)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_title('PESQ')
    axs.set_xticklabels(['-5', '0', '5', '10', '15', '20'])
    axs.set_xticks(np.arange(len(snrs_used)))
    axs.plot(mean_filtered_pesq, label='Filtrada', marker='o', linestyle='dashed')
    axs.plot(mean_noisy_pesq, label='Ruidosa', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    axs.set_xlabel('SNR [dB]')
    fig.savefig(f'./store/{filter_type}_{dataset_dir_name}_pesq_by_snr.png')
    plt.close(fig)

    mean_noisy_stoi = [np.asarray(noisy_stoi[snr]).mean() for snr in snrs_used]
    mean_filtered_stoi = [np.asarray(filtered_stoi[snr]).mean() for snr in snrs_used]

    with open(f'./store/{filter_type}_{dataset_dir_name}_noisy_stoi_by_snr.npy', 'wb') as f:
        np.save(f, mean_noisy_stoi)

    with open(f'./store/{filter_type}_{dataset_dir_name}_filtered_stoi_by_snr.npy', 'wb') as f:
        np.save(f, mean_filtered_stoi)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_title('STOI')
    axs.set_xticklabels(['-5', '0', '5', '10', '15', '20'])
    axs.set_xticks(np.arange(len(snrs_used)))
    axs.plot(mean_filtered_stoi, label='Filtrada', marker='o', linestyle='dashed')
    axs.plot(mean_noisy_stoi, label='Ruidosa', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    axs.set_xlabel('SNR [dB]')
    fig.savefig(f'./store/{filter_type}_{dataset_dir_name}_stoi_by_snr.png')
    plt.close(fig)

    mean_noisy_pesq = [np.asarray(noisy_pesq_by_noise_type[noise_type]).mean() for noise_type in noises_used]
    mean_filtered_pesq = [np.asarray(filtered_pesq_by_noise_type[noise_type]).mean() for noise_type in noises_used]

    with open(f'./store/{filter_type}_{dataset_dir_name}_noisy_pesq_by_noise_type.npy', 'wb') as f:
        np.save(f, mean_noisy_pesq)

    with open(f'./store/{filter_type}_{dataset_dir_name}_filtered_pesq_by_noise_type.npy', 'wb') as f:
        np.save(f, mean_filtered_pesq)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_title('PESQ')
    axs.set_xticklabels(
        ['Tipeo', 'Conversación', 'Vecinos', 'Aire acondicionado', 'Aspiradora', 'Impresora', 'Masticar'],
        rotation=45, ha='right'
    )
    axs.set_xticks(np.arange(len(noises_used)))
    axs.plot(mean_filtered_pesq, label='Filtrada', marker='o', linestyle='dashed')
    axs.plot(mean_noisy_pesq, label='Ruidosa', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    fig.savefig(f'./store/{filter_type}_{dataset_dir_name}_pesq_by_noise_type.png')
    plt.close(fig)

    mean_noisy_stoi = [np.asarray(noisy_stoi_by_noise_type[noise_type]).mean() for noise_type in noises_used]
    mean_filtered_stoi = [np.asarray(filtered_stoi_by_noise_type[noise_type]).mean() for noise_type in noises_used]

    with open(f'./store/{filter_type}_{dataset_dir_name}_noisy_stoi_by_noise_type.npy', 'wb') as f:
        np.save(f, mean_noisy_stoi)

    with open(f'./store/{filter_type}_{dataset_dir_name}_filtered_stoi_by_noise_type.npy', 'wb') as f:
        np.save(f, mean_filtered_stoi)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_title('STOI')
    axs.set_xticklabels(
        ['Tipeo', 'Conversación', 'Vecinos', 'Aire acondicionado', 'Aspiradora', 'Impresora', 'Masticar'],
        rotation=45, ha='right'
    )
    axs.set_xticks(np.arange(len(noises_used)))
    axs.plot(mean_filtered_stoi, label='Filtrada', marker='o', linestyle='dashed')
    axs.plot(mean_noisy_stoi, label='Ruidosa', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    fig.savefig(f'./store/{filter_type}_{dataset_dir_name}_stoi_by_noise_type.png')
    plt.close(fig)


@entry_point.command()
def compare_pesq_stoi_results():
    with open(f'./store/dnn_dataset_filtered_pesq_by_snr.npy', 'rb') as f:
        dnn_filtered_pesq_by_snr = np.load(f)

    with open(f'./store/dnn_dataset_filtered_stoi_by_snr.npy', 'rb') as f:
        dnn_filtered_stoi_by_snr = np.load(f)

    with open(f'./store/dnn_dataset_with_noise_type_filtered_pesq_by_noise_type.npy', 'rb') as f:
        dnn_filtered_pesq_by_noise_type = np.load(f)

    with open(f'./store/dnn_dataset_with_noise_type_filtered_stoi_by_noise_type.npy', 'rb') as f:
        dnn_filtered_stoi_by_noise_type = np.load(f)

    with open(f'./store/af_dataset_filtered_pesq_by_snr.npy', 'rb') as f:
        af_filtered_pesq_by_snr = np.load(f)

    with open(f'./store/af_dataset_filtered_stoi_by_snr.npy', 'rb') as f:
        af_filtered_stoi_by_snr = np.load(f)

    with open(f'./store/af_dataset_with_noise_type_filtered_pesq_by_noise_type.npy', 'rb') as f:
        af_filtered_pesq_by_noise_type = np.load(f)

    with open(f'./store/af_dataset_with_noise_type_filtered_stoi_by_noise_type.npy', 'rb') as f:
        af_filtered_stoi_by_noise_type = np.load(f)

    snrs_used = ['-5', '0', '5', '10', '15', '20']
    noises_used = ['Tipeo', 'Conversación', 'Vecinos', 'Aire acondicionado', 'Aspiradora', 'Impresora', 'Masticar']

    # By SNR
    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_xticklabels(snrs_used)
    axs.set_xticks(np.arange(len(snrs_used)))
    axs.plot(af_filtered_pesq_by_snr, label=r'Filtro adaptativo PESQ - Filtrada', marker='o', linestyle='dashed')
    axs.plot(dnn_filtered_pesq_by_snr, label=r'Filtro neuronal PESQ - Filtrada', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    axs.set_title('Comparación PESQ por SNR')
    axs.set_xlabel('SNR [dB]')
    fig.savefig('./store/comparison_pesq_by_snr.png')
    plt.close(fig)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_xticklabels(snrs_used)
    axs.set_xticks(np.arange(len(snrs_used)))
    axs.plot(af_filtered_stoi_by_snr, label=r'Filtro adaptativo STOI - Filtrada', marker='o', linestyle='dashed')
    axs.plot(dnn_filtered_stoi_by_snr, label=r'Filtro neuronal STOI - Filtrada', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    axs.set_title('Comparación STOI por SNR')
    axs.set_xlabel('SNR [dB]')
    fig.savefig('./store/comparison_stoi_by_snr.png')
    plt.close(fig)

    # By Noise
    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_xticklabels(noises_used, rotation=45, ha='right')
    axs.set_xticks(np.arange(len(noises_used)))
    axs.plot(af_filtered_pesq_by_noise_type, label=r'Filtro adaptativo PESQ - Filtrada', marker='o', linestyle='dashed')
    axs.plot(dnn_filtered_pesq_by_noise_type, label=r'Filtro neuronal PESQ - Filtrada', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    axs.set_title('Comparación PESQ por tipo de ruido')
    fig.savefig('./store/comparison_pesq_by_noise_type.png')
    plt.close(fig)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_xticklabels(noises_used, rotation=45, ha='right')
    axs.set_xticks(np.arange(len(noises_used)))
    axs.plot(af_filtered_stoi_by_noise_type, label=r'Filtro adaptativo STOI - Filtrada', marker='o', linestyle='dashed')
    axs.plot(dnn_filtered_stoi_by_noise_type, label=r'Filtro neuronal STOI - Filtrada', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    axs.set_title('Comparación STOI por tipo de ruido')
    fig.savefig('./store/comparison_stoi_by_noise_type.png')
    plt.close(fig)


if __name__ == '__main__':
    entry_point()
