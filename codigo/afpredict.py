import os
from os.path import join

import click
import torch

from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from adaptivealgorithms import rls
from afmetrics import AdaptiveFilteringMetricsEvaluator
import runsettings
from adaptivefilterdatasethandler import AdaptiveFilteringDataset

criterion = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on {}'.format(device))


@click.command()
@click.option('--output-dir', prompt='Output dir for trained models and event logs (gs prefix supported)', type=str)
@click.option('--input-dir', prompt='Dataset to be used while training', type=str)
@click.option('--experiment-name', prompt='Experiment name', type=str)
def predict(output_dir, input_dir, experiment_name):
    logs_gs_path = 'logs'

    experiment_id = experiment_name
    event_logs_dir = join(output_dir, logs_gs_path, experiment_id)

    dataset_path = input_dir

    print('Output tensorboard events to {}'.format(event_logs_dir))
    writer = SummaryWriter(log_dir=event_logs_dir, flush_secs=10)

    print('Preparing to predict')
    constant_filter = False
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    test_dataset = AdaptiveFilteringDataset(
        dataset_path, test_dataset_path, runsettings.fs, runsettings.af_windows_time_size,
        runsettings.overlap_percentage, randomize=runsettings.test_randomize_data,
        max_samples=runsettings.test_on_n_samples, constant_filter=constant_filter
    )
    test_data_loader = DataLoader(test_dataset, batch_size=runsettings.af_batch_size)

    print('Starting testing')

    test_metric_evaluator = AdaptiveFilteringMetricsEvaluator(
        writer, test_dataset, 'test', runsettings.fs
    )
    try:
        for i_batch, sample_batched in enumerate(test_data_loader):
            noisy_speech = sample_batched['noisy_speech'].numpy()[0, :]
            correlated_noise = sample_batched['correlated_noise'].numpy()[0, :]
            samples_idx = sample_batched['sample_idx'].numpy()[0]

            noise_estimation, filtered_speech, weights_n, _, _ = rls(
                correlated_noise, noisy_speech, runsettings.filter_size, lmda=runsettings.forgetting_rate
            )
            accumulated_samples_idx = test_dataset.accumulate_filtered_speech(filtered_speech, samples_idx)
            test_metric_evaluator.save_metrics(
                accumulated_samples_idx
            )
    except KeyboardInterrupt as e:
        writer.flush()
        raise e


if __name__ == "__main__":
    predict()
