import os
from typing import List

import click
from matplotlib import pyplot as plt, rc
from tensorflow.python.summary.summary_iterator import summary_iterator

rc('text', usetex=True)


@click.group()
def entry_point():
    pass


@entry_point.command()
@click.option('--output-dir', prompt='Where logs are located', type=str)
@click.option('--experiment-names', prompt='Experiment names separated by comma', type=str)
def analyze_train_data(output_dir, experiment_names):
    experiment_names = experiment_names.split(',')
    experiment_names = [experiment_name.strip() for experiment_name in experiment_names]

    train_mse_loss, test_mse_loss = [], []
    train_steps, test_steps = [], []

    for experiment_name in experiment_names:
        event_files_path = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(output_dir, experiment_name)):
            event_files_path.extend([os.path.join(dirpath, filename) for filename in filenames])

        tag_data, steps_data = get_tags_from_logs(
            [
                'MSE/train', 'MSE/test'

            ],
            ['train', 'test'],
            event_files_path
        )

        train_mse_loss.extend(tag_data['MSE/train'])
        train_steps.extend(steps_data['train'][:len(tag_data['MSE/train'])])

        test_mse_loss.extend(tag_data['MSE/test'])
        test_steps.extend(steps_data['test'][:len(tag_data['MSE/test'])])

    stop_step = 50000
    stop_step_test_index = test_steps.index(stop_step)
    stop_step_train_index = train_steps.index(stop_step)
    plot_mse_loss(
        train_steps[:stop_step_train_index], train_mse_loss[:stop_step_train_index],
        'ECM - Entrenamiento', 'train_mse', 0.9
    )
    plot_mse_loss(
        test_steps[:stop_step_test_index], test_mse_loss[:stop_step_test_index],
        'ECM - Validación', 'test_mse', 0.6
    )


def plot_mse_loss(
        steps, mse_loss, title, file_name, weight
):
    fig, axs = prepare_plot(title, steps)
    gen_plot(steps, mse_loss, axs, 'ECM', weight, (1, 0, 0))
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
    # axs.annotate(
    #     text='{:.1e}'.format(ema[-1]),
    #     xy=(steps[-1], ema[-1]),
    #     horizontalalignment='center',
    #     verticalalignment='center',
    #     textcoords='offset pixels',
    #     xytext=(50, 0),
    #     fontsize=16
    # )


def get_ema(scalars, weight):
    last = scalars[0]
    ema = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        ema.append(smoothed_val)
        last = smoothed_val

    return ema


def get_tags_from_logs(tags: List[str], step_tags: List[str], event_files_path: List[bytes]):
    data = {tag: [] for tag in tags}
    steps_data = {step_tag: [] for step_tag in step_tags}
    for event_file_path in event_files_path:
        for event in summary_iterator(event_file_path):
            for value in event.summary.value:
                for step_tag in step_tags:
                    if step_tag in value.tag: # and event.step not in steps_data[step_tag]:
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


# @entry_point.command()
# @click.option('--output-dir', prompt='Where logs are located', type=str)
# @click.option('--af-experiment-name', prompt='Experiment name', type=str)
# @click.option('--dnn-experiment-name', prompt='Experiment name', type=str)
# def compare_results(output_dir, af_experiment_name, dnn_experiment_name):
#     (
#         af_min_pesq_by_snr, af_pesq_by_snr, af_min_stoi_by_snr, af_stoi_by_snr, snrs_used, _, _, _, _, _, _, _, _,
#         af_pesq_by_noise, af_min_pesq_by_noise, af_stoi_by_noise, af_min_stoi_by_noise, noised_used
#     ) = (
#         _analyze(output_dir, af_experiment_name)
#     )
#
#     (
#         dnn_min_pesq_by_snr, dnn_pesq_by_snr, dnn_min_stoi_by_snr, dnn_stoi_by_snr, snrs_used, _, _, _, _, _, _, _, _,
#         dnn_pesq_by_noise, dnn_min_pesq_by_noise, dnn_stoi_by_noise, dnn_min_stoi_by_noise, noised_used
#     ) = (
#         _analyze(output_dir, dnn_experiment_name)
#     )
#
#     # By SNR
#     af_delta_mean_pesq = compute_deltas(snrs_used, af_pesq_by_snr, af_min_pesq_by_snr)
#     af_delta_mean_stoi = compute_deltas(snrs_used, af_stoi_by_snr, af_min_stoi_by_snr)
#     dnn_delta_mean_pesq = compute_deltas(snrs_used, dnn_pesq_by_snr, dnn_min_pesq_by_snr)
#     dnn_delta_mean_stoi = compute_deltas(snrs_used, dnn_stoi_by_snr, dnn_min_stoi_by_snr)
#
#     fig, axs = plt.subplots(dpi=100, tight_layout=True)
#     axs.set_xticklabels(['-5 dB', '0 dB', '5 dB', '10 dB', '15 dB', '20 dB', r'$\infty$ dB'])
#     axs.set_xticks(np.arange(len(snrs_used)))
#     axs.plot(af_delta_mean_pesq, label=r'Filtro adaptativo $\Delta$ PESQ', marker='o', linestyle='dashed')
#     axs.plot(dnn_delta_mean_pesq, label=r'Filtro neuronal $\Delta$ PESQ', marker='o', linestyle='dashed')
#     axs.legend()
#     axs.grid()
#     fig.savefig('./store/comparison_pesq.png')
#     plt.close(fig)
#
#     fig, axs = plt.subplots(dpi=100, tight_layout=True)
#     axs.set_xticklabels(['-5 dB', '0 dB', '5 dB', '10 dB', '15 dB', '20 dB', r'$\infty$ dB'])
#     axs.set_xticks(np.arange(len(snrs_used)))
#     axs.plot(af_delta_mean_stoi, label=r'Filtro adaptativo $\Delta$ STOI', marker='o', linestyle='dashed')
#     axs.plot(dnn_delta_mean_stoi, label=r'Filtro neuronal $\Delta$ STOI', marker='o', linestyle='dashed')
#     axs.legend()
#     axs.grid()
#     fig.savefig('./store/comparison_stoi.png')
#     plt.close(fig)
#
#     # By Noise
#     af_delta_mean_pesq = compute_deltas(noised_used, af_pesq_by_noise, af_min_pesq_by_noise)
#     af_delta_mean_stoi = compute_deltas(noised_used, af_stoi_by_noise, af_min_stoi_by_noise)
#     dnn_delta_mean_pesq = compute_deltas(noised_used, dnn_pesq_by_noise, dnn_min_pesq_by_noise)
#     dnn_delta_mean_stoi = compute_deltas(noised_used, dnn_stoi_by_noise, dnn_min_stoi_by_noise)
#
#     fig, axs = plt.subplots(dpi=100, tight_layout=True)
#     axs.set_xticklabels(['Ladrido', 'Maullido', 'Trafico', 'Tipeo', 'Conversación'])
#     axs.set_xticks(np.arange(len(snrs_used)))
#     axs.plot(af_delta_mean_pesq, label=r'Filtro adaptativo $\Delta$ PESQ', marker='o', linestyle='dashed')
#     axs.plot(dnn_delta_mean_pesq, label=r'Filtro neuronal $\Delta$ PESQ', marker='o', linestyle='dashed')
#     axs.legend()
#     axs.grid()
#     fig.savefig('./store/comparison_pesq.png')
#     plt.close(fig)
#
#     fig, axs = plt.subplots(dpi=100, tight_layout=True)
#     axs.set_xticklabels(['Ladrido', 'Maullido', 'Trafico', 'Tipeo', 'Conversación'])
#     axs.set_xticks(np.arange(len(snrs_used)))
#     axs.plot(af_delta_mean_stoi, label=r'Filtro adaptativo $\Delta$ STOI', marker='o', linestyle='dashed')
#     axs.plot(dnn_delta_mean_stoi, label=r'Filtro neuronal $\Delta$ STOI', marker='o', linestyle='dashed')
#     axs.legend()
#     axs.grid()
#     fig.savefig('./store/comparison_stoi.png')
#     plt.close(fig)
#
#
# def compute_deltas(snrs_or_noises_used, metric_by_snr_or_noise, min_metric_by_snr_or_noise):
#     delta_mean_metrics = []
#     for snr in snrs_or_noises_used:
#         metric = np.asarray(metric_by_snr_or_noise[snr])
#         min_metric = np.asarray(min_metric_by_snr_or_noise[snr])
#         delta = metric - min_metric
#
#         moving_average_den = (np.arange(len(metric)) + 1)
#         delta_moving_average = np.cumsum(delta) / moving_average_den
#         delta_mean_metrics.append(delta_moving_average[-1])
#
#     return delta_mean_metrics


if __name__ == '__main__':
    entry_point()
