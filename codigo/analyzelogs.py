import os
import statistics
from typing import List

import click
import numpy as np
from matplotlib import pyplot as plt, rc
from tensorflow.python.summary.summary_iterator import summary_iterator

rc('text', usetex=True)


@click.group()
def entry_point():
    pass


@entry_point.command()
@click.option('--output-dir', prompt='Where logs are located', type=str)
@click.option('--experiment-name', prompt='Experiment name', type=str)
def analyze(output_dir, experiment_name):
    (
        min_pesq_by_snr, pesq_by_snr, min_stoi_by_snr, stoi_by_snr, snrs_used, audio_max_mse_loss, audio_mse_loss,
        max_mse_by_snr, mse_by_snr, pesq_by_snr_and_noise, min_pesq_by_snr_and_noise,
        stoi_by_snr_and_noise, min_stoi_by_snr_and_noise, pesq_by_noise, min_pesq_by_noise, stoi_by_noise,
        min_stoi_by_noise, noised_used
    ) = (
        _analyze(output_dir, experiment_name)
    )

    plot_objective_metric(min_pesq_by_snr, pesq_by_snr, snrs_used, 'PESQ')
    plot_objective_metric(min_stoi_by_snr, stoi_by_snr, snrs_used, 'STOI')
    plot_objective_metric(max_mse_by_snr, mse_by_snr, snrs_used, 'MSE', metric_is_an_error=True)
    if 'af-' in experiment_name:
        plot_audio_mse(audio_max_mse_loss, audio_mse_loss)

    plot_objective_metric(min_pesq_by_noise, pesq_by_noise, noised_used, 'PESQ', is_snr=False)
    plot_objective_metric(min_stoi_by_noise, stoi_by_noise, noised_used, 'STOI', is_snr=False)


def _analyze(output_dir, experiment_name):
    event_files_path = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(output_dir, 'logs', experiment_name)):
        event_files_path.extend([os.path.join(dirpath, filename) for filename in filenames])

    snrs_used = ['-5db', '0db', '5db', '10db', '15db', '20db', 'infdb']
    noised_used = ['bark', 'meow', 'traffic', 'typing', 'babble']

    tags = [
        [
            f"MSE_SNR_{snr}/test.actual", f"MSE_SNR_{snr}/test.max",
            f"PESQ_SNR_{snr}/test.actual", f"PESQ_SNR_{snr}/test.min",
            f"STOI_SNR_{snr}/test.actual", f"STOI_SNR_{snr}/test.min",
            list([
                     f'MSE_SNR_{snr}_NOISE_{noise}/test.actual', f'MSE_SNR_{snr}_NOISE_{noise}/test.max',
                     f'PESQ_SNR_{snr}_NOISE_{noise}/test.actual', f'PESQ_SNR_{snr}_NOISE_{noise}/test.min',
                     f'STOI_SNR_{snr}_NOISE_{noise}/test.actual', f'STOI_SNR_{snr}_NOISE_{noise}/test.min',
                 ] for noise in noised_used),
        ] for snr in snrs_used
    ]

    def flatten(l):
        return sum(map(flatten, l), []) if isinstance(l, list) else [l]

    tags = flatten(tags)
    tags = tags + ['AudioMSELoss/test.actual', 'AudioMSELoss/test.max']

    tag_data, steps_data = get_tags_from_logs(
        tags,
        ['test'],
        event_files_path
    )

    pesq_by_snr = {
        snr: [] for snr in snrs_used
    }
    min_pesq_by_snr = {
        snr: [] for snr in snrs_used
    }

    pesq_by_snr_and_noise = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }
    min_pesq_by_snr_and_noise = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }

    pesq_by_noise = {
        noise_type: [] for noise_type in noised_used
    }
    min_pesq_by_noise = {
        noise_type: [] for noise_type in noised_used
    }

    stoi_by_snr = {
        snr: [] for snr in snrs_used
    }
    min_stoi_by_snr = {
        snr: [] for snr in snrs_used
    }

    stoi_by_snr_and_noise = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }
    min_stoi_by_snr_and_noise = {
        snr: {noise_type: [] for noise_type in noised_used} for snr in snrs_used
    }

    stoi_by_noise = {
        noise_type: [] for noise_type in noised_used
    }
    min_stoi_by_noise = {
        noise_type: [] for noise_type in noised_used
    }

    mse_by_snr = {
        snr: [] for snr in snrs_used
    }
    max_mse_by_snr = {
        snr: [] for snr in snrs_used
    }

    audio_mse_loss = []
    audio_max_mse_loss = []

    for key in tag_data.keys():
        if 'AudioMSELoss' in key and 'actual' in key:
            audio_mse_loss = tag_data[key]
        elif 'AudioMSELoss' in key and 'max' in key:
            audio_max_mse_loss = tag_data[key]
        elif 'MSE_SNR' in key and 'actual' in key:
            if 'NOISE' not in key:
                start = key.index('SNR_') + 4
                end = key.index('/')
                mse_by_snr[key[start:end]] = tag_data[key]
        elif 'MSE_SNR' in key and 'max' in key:
            if 'NOISE' not in key:
                start = key.index('SNR_') + 4
                end = key.index('/')
                max_mse_by_snr[key[start:end]] = tag_data[key]
        elif 'PESQ_SNR' in key and 'actual' in key:
            if 'NOISE' not in key:
                snr_start = key.index('SNR_') + 4
                snr_end = key.index('/')
                pesq_by_snr[key[snr_start:snr_end]] = tag_data[key]
            else:
                snr_start = key.index('SNR_') + 4
                snr_end = key.index('_NOISE_')
                noise_start = key.index('_NOISE_') + 7
                noise_end = key.index('/')
                pesq_by_snr_and_noise[key[snr_start:snr_end]][key[noise_start:noise_end]] = tag_data[key]
                pesq_by_noise[key[noise_start:noise_end]].extend(tag_data[key])
        elif 'PESQ_SNR' in key and 'min' in key:
            if 'NOISE' not in key:
                start = key.index('SNR_') + 4
                end = key.index('/')
                min_pesq_by_snr[key[start:end]] = tag_data[key]
            else:
                snr_start = key.index('SNR_') + 4
                snr_end = key.index('_NOISE_')
                noise_start = key.index('_NOISE_') + 7
                noise_end = key.index('/')
                min_pesq_by_snr_and_noise[key[snr_start:snr_end]][key[noise_start:noise_end]] = tag_data[key]
                min_pesq_by_noise[key[noise_start:noise_end]].extend(tag_data[key])
        elif 'STOI_SNR' in key and 'actual' in key:
            if 'NOISE' not in key:
                start = key.index('SNR_') + 4
                end = key.index('/')
                stoi_by_snr[key[start:end]] = tag_data[key]
            else:
                snr_start = key.index('SNR_') + 4
                snr_end = key.index('_NOISE_')
                noise_start = key.index('_NOISE_') + 7
                noise_end = key.index('/')
                stoi_by_snr_and_noise[key[snr_start:snr_end]][key[noise_start:noise_end]] = tag_data[key]
                stoi_by_noise[key[noise_start:noise_end]].extend(tag_data[key])
        elif 'STOI_SNR' in key and 'min' in key:
            if 'NOISE' not in key:
                start = key.index('SNR_') + 4
                end = key.index('/')
                min_stoi_by_snr[key[start:end]] = tag_data[key]
            else:
                snr_start = key.index('SNR_') + 4
                snr_end = key.index('_NOISE_')
                noise_start = key.index('_NOISE_') + 7
                noise_end = key.index('/')
                min_stoi_by_snr_and_noise[key[snr_start:snr_end]][key[noise_start:noise_end]] = tag_data[key]
                min_stoi_by_noise[key[noise_start:noise_end]].extend(tag_data[key])

    return (
        min_pesq_by_snr, pesq_by_snr, min_stoi_by_snr, stoi_by_snr, snrs_used, audio_max_mse_loss, audio_mse_loss,
        max_mse_by_snr, mse_by_snr, pesq_by_snr_and_noise, min_pesq_by_snr_and_noise,
        stoi_by_snr_and_noise, min_stoi_by_snr_and_noise, pesq_by_noise, min_pesq_by_noise, stoi_by_noise,
        min_stoi_by_noise, noised_used
    )


def plot_audio_mse(
        audio_max_mse_loss, audio_mse_loss
):
    audio_max_mse_loss = np.asarray(audio_max_mse_loss)
    audio_mse_loss = np.asarray(audio_mse_loss)
    delta = audio_max_mse_loss - audio_mse_loss

    moving_average_den = (np.arange(len(audio_mse_loss)) + 1)
    moving_average = np.cumsum(audio_mse_loss) / moving_average_den
    max_moving_average = np.cumsum(audio_max_mse_loss) / moving_average_den
    delta_moving_average = np.cumsum(delta) / moving_average_den

    x_axis = np.arange(len(audio_mse_loss)) * 50
    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.ticklabel_format(axis='y', style='scientific')
    axs.set_xlim(right=x_axis[-1] + 0.2 * x_axis[-1])
    axs.plot(x_axis, audio_max_mse_loss, color=(0, 0, 1, 0.3), label='Max Audio MSE')
    axs.plot(
        x_axis, max_moving_average, color=(0, 0, 1, 1), marker='o', markevery=[len(audio_mse_loss) - 1],
        label='Max Audio MSE - Valor medio'
    )
    axs.annotate(
        text='{:.2e}'.format(max_moving_average[-1]),
        xy=(x_axis[-1], max_moving_average[-1]),
        horizontalalignment='center',
        verticalalignment='center',
        textcoords='offset pixels',
        xytext=(50, 0),
        fontsize=16
    )
    axs.plot(x_axis, audio_mse_loss, color=(1, 0, 0, 0.3), label='Audio MSE')
    axs.plot(
        x_axis, moving_average, color=(1, 0, 0, 1), label='Audio MSE - Valor medio',
        marker='o', markevery=[len(moving_average) - 1]
    )
    axs.annotate(
        text='{:.2e}'.format(moving_average[-1]),
        xy=(x_axis[-1], moving_average[-1]),
        horizontalalignment='center',
        verticalalignment='center',
        textcoords='offset pixels',
        xytext=(50, 0),
        fontsize=16
    )
    axs.plot(x_axis, delta, color=(19 / 255, 153 / 255, 19 / 255, 0.3), label=r'$\Delta$ Audio MSE ')
    axs.plot(
        x_axis, delta_moving_average, color=(19 / 255, 153 / 255, 19 / 255, 1),
        label=r'$\Delta$ Audio MSE - Valor medio',
        marker='o', markevery=[len(audio_mse_loss) - 1]
    )
    axs.annotate(
        text='{:.2e}'.format(delta_moving_average[-1]),
        xy=(x_axis[-1], delta_moving_average[-1]),
        horizontalalignment='center',
        verticalalignment='center',
        textcoords='offset pixels',
        xytext=(50, 0),
        fontsize=16
    )
    axs.grid()
    axs.legend()
    fig.savefig('./store/audio_mse.png')
    plt.close(fig)


def plot_objective_metric(
        bound_metric_by_snr_or_noise, metric_by_snr_or_noise,
        snrs_or_noises_used, metric_name, metric_is_an_error=False,
        is_snr=True
):
    mean_metrics = []
    bound_mean_metrics = []
    delta_mean_metrics = []

    if metric_is_an_error:
        bound_str = 'Max'
    else:
        bound_str = 'Min'

    for snr_or_noise in snrs_or_noises_used:
        metric = np.asarray(metric_by_snr_or_noise[snr_or_noise])
        bound_metric = np.asarray(bound_metric_by_snr_or_noise[snr_or_noise])
        if metric_is_an_error:
            delta = bound_metric - metric
        else:
            delta = metric - bound_metric

        moving_average_den = (np.arange(len(metric)) + 1)
        moving_average = np.cumsum(metric) / moving_average_den
        bound_moving_average = np.cumsum(bound_metric) / moving_average_den
        delta_moving_average = np.cumsum(delta) / moving_average_den

        mean_metrics.append(moving_average[-1])
        bound_mean_metrics.append(bound_moving_average[-1])
        delta_mean_metrics.append(delta_moving_average[-1])

        x_axis = np.arange(len(metric))
        fig, axs = plt.subplots(dpi=100, tight_layout=True)
        axs.grid()
        axs.set_xlim(right=x_axis[-1] + 0.1 * x_axis[-1])
        axs.plot(x_axis, bound_metric, color=(0, 0, 1, 0.3), label='{} {}'.format(bound_str, metric_name))
        axs.plot(
            x_axis, bound_moving_average, color=(0, 0, 1, 1), marker='o', markevery=[len(metric) - 1],
            label='{} {} - Valor medio'.format(bound_str, metric_name)
        )
        axs.annotate(
            text='{:.2f}'.format(bound_moving_average[-1]),
            xy=(x_axis[-1], bound_moving_average[-1]),
            horizontalalignment='center',
            verticalalignment='center',
            textcoords='offset pixels',
            xytext=(30, 0),
            fontsize=16
        )
        axs.plot(x_axis, metric, color=(1, 0, 0, 0.3), label=metric_name)
        axs.plot(
            x_axis, moving_average, color=(1, 0, 0, 1), label='{} - Valor medio'.format(metric_name),
            marker='o', markevery=[len(metric) - 1]
        )
        axs.annotate(
            text='{:.2f}'.format(moving_average[-1]),
            xy=(x_axis[-1], moving_average[-1]),
            horizontalalignment='center',
            verticalalignment='center',
            textcoords='offset pixels',
            xytext=(30, 0),
            fontsize=16
        )
        axs.plot(x_axis, delta, color=(19 / 255, 153 / 255, 19 / 255, 0.3), label=r'$\Delta$ {} '.format(metric_name))
        axs.plot(
            x_axis, delta_moving_average, color=(19 / 255, 153 / 255, 19 / 255, 1),
            label=r'$\Delta$ {} - Valor medio'.format(metric_name),
            marker='o', markevery=[len(metric) - 1]
        )
        axs.annotate(
            text='{:.2f}'.format(delta_moving_average[-1]),
            xy=(x_axis[-1], delta_moving_average[-1]),
            horizontalalignment='center',
            verticalalignment='center',
            textcoords='offset pixels',
            xytext=(30, 0),
            fontsize=16
        )
        axs.legend()
        fig.savefig('./store/metric_{}_{}.png'.format(metric_name, snr_or_noise))
        plt.close(fig)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    if is_snr:
        axs.set_xticklabels(['-5 dB', '0 dB', '5 dB', '10 dB', '15 dB', '20 dB', r'$\infty$ dB'])
    else:
        axs.set_xticklabels(['Ladrido', 'Maullido', 'Trafico', 'Tipeo', 'Conversación'])
    axs.set_xticks(np.arange(len(snrs_or_noises_used)))
    axs.plot(mean_metrics, label='{}'.format(metric_name), marker='o', linestyle='dashed')
    axs.plot(bound_mean_metrics, label='{} {}'.format(bound_str, metric_name), marker='o', linestyle='dashed')
    axs.plot(delta_mean_metrics, label=r'$\Delta$ {}'.format(metric_name), marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    fig.savefig('./store/metric_{}.png'.format(metric_name))
    plt.close(fig)


def print_mean_values(
        min_pesq_by_snr_and_noise_type, pesq_by_snr_and_noise_type, min_pesq_by_snr, pesq_by_snr,
        min_stoi_by_snr_and_noise_type, stoi_by_snr_and_noise_type, min_stoi_by_snr, stoi_by_snr,
        snrs_used, noised_used
):
    text_lines = []
    for snr in snrs_used:
        text_values = []
        for noise in noised_used:
            min_mean = statistics.mean(min_pesq_by_snr_and_noise_type[snr][noise])
            actual_mean = statistics.mean(pesq_by_snr_and_noise_type[snr][noise])
            delta = ((actual_mean - min_mean) / 4) * 100
            text_values.append('{:.2f}\t{:.2f}\t{}'.format(
                round(min_mean, 2),
                round(actual_mean, 2),
                round(delta),
            ))

        min_mean = statistics.mean(min_pesq_by_snr[snr])
        actual_mean = statistics.mean(pesq_by_snr[snr])
        delta = ((actual_mean - min_mean) / 4) * 100
        text_values.append('{:.2f}\t{:.2f}\t{}'.format(
            round(min_mean, 2),
            round(actual_mean, 2),
            round(delta),
        ))

        text_lines.append('\t'.join(text_values))

    print('\n'.join(text_lines))

    text_lines = []
    for snr in snrs_used:
        text_values = []
        for noise in noised_used:
            min_mean = statistics.mean(min_stoi_by_snr_and_noise_type[snr][noise])
            actual_mean = statistics.mean(stoi_by_snr_and_noise_type[snr][noise])
            delta = ((actual_mean - min_mean) / 1) * 100
            text_values.append('{:.2f}\t\t{:.2f}\t\t{}'.format(
                round(min_mean, 2),
                round(actual_mean, 2),
                round(delta),
            ))

        min_mean = statistics.mean(min_stoi_by_snr[snr])
        actual_mean = statistics.mean(stoi_by_snr[snr])
        delta = ((actual_mean - min_mean) / 1) * 100
        text_values.append('{:.2f}\t\t{:.2f}\t\t{}'.format(
            round(min_mean, 2),
            round(actual_mean, 2),
            round(delta),
        ))

        text_lines.append('\t'.join(text_values))

    print('\n'.join(text_lines))


@entry_point.command()
@click.option('--output-dir', prompt='Where logs are located', type=str)
@click.option('--experiment-name', prompt='Experiment name', type=str)
def analyze_train_data(output_dir, experiment_name):
    event_files_path = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(output_dir, 'logs', experiment_name)):
        event_files_path.extend([os.path.join(dirpath, filename) for filename in filenames])

    tag_data, steps_data = get_tags_from_logs(
        [
            'MSELoss/train', 'MSELoss/val', 'STFTMapeLoss/train.actual', 'STFTMapeLoss/train.max',
            'STFTMapeLoss/val.actual', 'STFTMapeLoss/val.max', 'TimeDomainMapeLoss/train.actual',
            'TimeDomainMapeLoss/train.max', 'TimeDomainMapeLoss/val.actual', 'TimeDomainMapeLoss/val.max'

        ],
        ['train', 'val'],
        event_files_path
    )

    train_mse_loss, val_mse_loss = tag_data['MSELoss/train'], tag_data['MSELoss/val']
    train_stft_mape_loss, train_stft_max_mape_loss = (
        tag_data['STFTMapeLoss/train.actual'], tag_data['STFTMapeLoss/train.max']
    )
    val_stft_mape_loss, val_stft_max_mape_loss = tag_data['STFTMapeLoss/val.actual'], tag_data['STFTMapeLoss/val.max']
    train_time_domain_mape_loss, train_time_domain_max_mape_loss = (
        tag_data['TimeDomainMapeLoss/train.actual'], tag_data['TimeDomainMapeLoss/train.max']
    )
    val_time_domain_mape_loss, val_time_domain_max_mape_loss = (
        tag_data['TimeDomainMapeLoss/val.actual'], tag_data['TimeDomainMapeLoss/val.max']
    )
    train_steps, val_steps = tag_data['train'], tag_data['val']

    plot_mse_loss(train_steps, train_mse_loss, 'ECM - Entrenamiento', 'train_mse', 0.9)
    plot_mse_loss(val_steps, val_mse_loss, 'ECM - Validación', 'val_mse', 0.6)

    plot_stft_mape_loss(
        train_steps, train_stft_mape_loss, train_stft_max_mape_loss, 'STFT MAPE - Entrenamiento', 'train_stft_mape', 0.9
    )
    plot_stft_mape_loss(
        val_steps, val_stft_mape_loss, val_stft_max_mape_loss, 'STFT MAPE - Validación', 'val_stft_mape', 0.6
    )

    plot_stft_mape_loss(
        train_steps, train_time_domain_mape_loss, train_time_domain_max_mape_loss, 'Audio MAPE - Entrenamiento',
        'train_audio_mape', 0.9
    )
    plot_stft_mape_loss(
        val_steps, val_time_domain_mape_loss, val_time_domain_max_mape_loss, 'Audio MAPE - Validación',
        'val_audio_mape', 0.6
    )


def plot_mse_loss(
        steps, mse_loss, title, file_name, weight
):
    fig, axs = prepare_plot(title, steps)
    gen_plot(steps, mse_loss, axs, 'ECM', weight, (1, 0, 0))
    axs.legend()
    fig.savefig(f'./store/{file_name}.png')
    plt.close(fig)


def plot_stft_mape_loss(steps, stft_mape_loss, max_stft_mape_loss, title, file_name, weight):
    fig, axs = prepare_plot(title, steps)
    gen_plot(steps, stft_mape_loss, axs, 'STFT MAPE', weight, (1, 0, 0))
    gen_plot(steps, max_stft_mape_loss, axs, 'STFT Max MAPE', weight, (0, 0, 1))
    axs.legend()
    fig.savefig(f'./store/{file_name}.png')
    plt.close(fig)


def prepare_plot(title, steps):
    fig, axs = plt.subplots(figsize=(10, 5), dpi=100, tight_layout=True)
    axs.ticklabel_format(axis='y', style='scientific')
    axs.set_xlim(right=steps[-1] + 1500)
    axs.set_yscale('log')
    axs.grid()
    axs.set_title(title)
    return fig, axs


def gen_plot(steps, measure, axs, label, weight, color):
    ema = get_ema(measure, weight)
    axs.plot(steps, measure, color=color + (0.2,), label=f'{label}')
    axs.plot(
        steps, ema, color=color + (1,), marker='o', markevery=[len(measure) - 1],
        label=f'{label} - Valor medio'
    )
    axs.annotate(
        text='{:.1e}'.format(ema[-1]),
        xy=(steps[-1], ema[-1]),
        horizontalalignment='center',
        verticalalignment='center',
        textcoords='offset pixels',
        xytext=(50, 0),
        fontsize=16
    )


def get_ema(scalars, weight):
    last = scalars[0]
    ema = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        ema.append(smoothed_val)
        last = smoothed_val

    return ema


def get_tags_from_logs(tags: List[str], step_tags: List[str], event_files_path: List[str]):
    data = {tag: [] for tag in tags}
    steps_data = {step_tag: [] for step_tag in step_tags}
    for event_file_path in event_files_path:
        for event in summary_iterator(event_file_path):
            for value in event.summary.value:
                for step_tag in step_tags:
                    if step_tag in value.tag and event.step not in steps_data[step_tag]:
                        steps_data[step_tag].append(event.step)

                for tag in tags:
                    metric_name_and_case = tag.split('.')
                    if len(metric_name_and_case) == 1:
                        metric_name, case = metric_name_and_case[0], None
                    else:
                        metric_name, case = metric_name_and_case

                    if value.tag == metric_name:
                        if (not case) or (case and case in event_file_path):
                            data[tag].append(value.simple_value)

    return data, steps_data


@entry_point.command()
@click.option('--output-dir', prompt='Where logs are located', type=str)
@click.option('--af-experiment-name', prompt='Experiment name', type=str)
@click.option('--dnn-experiment-name', prompt='Experiment name', type=str)
def compare_results(output_dir, af_experiment_name, dnn_experiment_name):
    (
        af_min_pesq_by_snr, af_pesq_by_snr, af_min_stoi_by_snr, af_stoi_by_snr, snrs_used, _, _, _, _, _, _, _, _,
        af_pesq_by_noise, af_min_pesq_by_noise, af_stoi_by_noise, af_min_stoi_by_noise, noised_used
    ) = (
        _analyze(output_dir, af_experiment_name)
    )

    (
        dnn_min_pesq_by_snr, dnn_pesq_by_snr, dnn_min_stoi_by_snr, dnn_stoi_by_snr, snrs_used, _, _, _, _, _, _, _, _,
        dnn_pesq_by_noise, dnn_min_pesq_by_noise, dnn_stoi_by_noise, dnn_min_stoi_by_noise, noised_used
    ) = (
        _analyze(output_dir, dnn_experiment_name)
    )

    # By SNR
    # af_delta_mean_pesq = compute_deltas(snrs_used, af_pesq_by_snr, af_min_pesq_by_snr)
    # af_delta_mean_stoi = compute_deltas(snrs_used, af_stoi_by_snr, af_min_stoi_by_snr)
    # dnn_delta_mean_pesq = compute_deltas(snrs_used, dnn_pesq_by_snr, dnn_min_pesq_by_snr)
    # dnn_delta_mean_stoi = compute_deltas(snrs_used, dnn_stoi_by_snr, dnn_min_stoi_by_snr)
    #
    # fig, axs = plt.subplots(dpi=100, tight_layout=True)
    # axs.set_xticklabels(['-5 dB', '0 dB', '5 dB', '10 dB', '15 dB', '20 dB', r'$\infty$ dB'])
    # axs.set_xticks(np.arange(len(snrs_used)))
    # axs.plot(af_delta_mean_pesq, label=r'Filtro adaptativo $\Delta$ PESQ', marker='o', linestyle='dashed')
    # axs.plot(dnn_delta_mean_pesq, label=r'Filtro neuronal $\Delta$ PESQ', marker='o', linestyle='dashed')
    # axs.legend()
    # axs.grid()
    # fig.savefig('./store/comparison_pesq.png')
    # plt.close(fig)
    #
    # fig, axs = plt.subplots(dpi=100, tight_layout=True)
    # axs.set_xticklabels(['-5 dB', '0 dB', '5 dB', '10 dB', '15 dB', '20 dB', r'$\infty$ dB'])
    # axs.set_xticks(np.arange(len(snrs_used)))
    # axs.plot(af_delta_mean_stoi, label=r'Filtro adaptativo $\Delta$ STOI', marker='o', linestyle='dashed')
    # axs.plot(dnn_delta_mean_stoi, label=r'Filtro neuronal $\Delta$ STOI', marker='o', linestyle='dashed')
    # axs.legend()
    # axs.grid()
    # fig.savefig('./store/comparison_stoi.png')
    # plt.close(fig)

    # By Noise
    af_delta_mean_pesq = compute_deltas(noised_used, af_pesq_by_noise, af_min_pesq_by_noise)
    af_delta_mean_stoi = compute_deltas(noised_used, af_stoi_by_noise, af_min_stoi_by_noise)
    dnn_delta_mean_pesq = compute_deltas(noised_used, dnn_pesq_by_noise, dnn_min_pesq_by_noise)
    dnn_delta_mean_stoi = compute_deltas(noised_used, dnn_stoi_by_noise, dnn_min_stoi_by_noise)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_xticklabels(['Ladrido', 'Maullido', 'Trafico', 'Tipeo', 'Conversación'])
    axs.set_xticks(np.arange(len(snrs_used)))
    axs.plot(af_delta_mean_pesq, label=r'Filtro adaptativo $\Delta$ PESQ', marker='o', linestyle='dashed')
    axs.plot(dnn_delta_mean_pesq, label=r'Filtro neuronal $\Delta$ PESQ', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    fig.savefig('./store/comparison_pesq.png')
    plt.close(fig)

    fig, axs = plt.subplots(dpi=100, tight_layout=True)
    axs.set_xticklabels(['Ladrido', 'Maullido', 'Trafico', 'Tipeo', 'Conversación'])
    axs.set_xticks(np.arange(len(snrs_used)))
    axs.plot(af_delta_mean_stoi, label=r'Filtro adaptativo $\Delta$ STOI', marker='o', linestyle='dashed')
    axs.plot(dnn_delta_mean_stoi, label=r'Filtro neuronal $\Delta$ STOI', marker='o', linestyle='dashed')
    axs.legend()
    axs.grid()
    fig.savefig('./store/comparison_stoi.png')
    plt.close(fig)


def compute_deltas(snrs_or_noises_used, metric_by_snr_or_noise, min_metric_by_snr_or_noise):
    delta_mean_metrics = []
    for snr in snrs_or_noises_used:
        metric = np.asarray(metric_by_snr_or_noise[snr])
        min_metric = np.asarray(min_metric_by_snr_or_noise[snr])
        delta = metric - min_metric

        moving_average_den = (np.arange(len(metric)) + 1)
        delta_moving_average = np.cumsum(delta) / moving_average_den
        delta_mean_metrics.append(delta_moving_average[-1])

    return delta_mean_metrics


if __name__ == '__main__':
    entry_point()
