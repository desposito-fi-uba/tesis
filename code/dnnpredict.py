import os
from os.path import isfile, join

import click
import torch
import numpy as np

from torch import nn, sigmoid
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from constants import FilterType
import runsettings
from dnnmetrics import MetricsEvaluator
from realtimednnfilterdatasethandler import RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures

criterion = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on {}'.format(device))

print('Running in mode {}'.format(runsettings.filter_type))
print('Running with features {}'.format(runsettings.features_type))
print('Running with optimizer {}'.format(runsettings.optimizer_type))


@click.command()
@click.option('--model', required=False, help='Name of the model to be used to predict', type=str)
@click.option('--output-dir', prompt='Output dir for trained models and event logs (gs prefix supported)', type=str)
@click.option('--input-dir', prompt='Dataset to be used while training', type=str)
@click.option('--experiment-name', prompt='Experiment name', type=str)
def predict(output_dir, input_dir, experiment_name, model=None):
    if model is None:
        models_path = os.path.join(os.getcwd(), 'trained-models')

        models = np.array([
            os.path.join(models_path, file_path)
            for file_path in os.listdir(models_path)
            if isfile(join(models_path, file_path))
        ])
        model_path = max(models, key=os.path.getctime)
    else:
        model_path = os.path.join(os.getcwd(), 'trained-models', model)

    logs_gs_path = 'logs'

    experiment_id = experiment_name
    event_logs_dir = join(output_dir, logs_gs_path, experiment_id)

    dataset_path = input_dir

    print('Model {} loaded'.format(model_path))
    runsettings.net.load_state_dict(torch.load(model_path))

    print('Output tensorboard events to {}'.format(event_logs_dir))
    writer = SummaryWriter(log_dir=event_logs_dir)

    print('Preparing to predict')
    test_dataset_path = os.path.join(dataset_path, 'test.csv')

    test_dataset = RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(
        dataset_path, test_dataset_path, runsettings.fs, runsettings.windows_time_size,
        runsettings.overlap_percentage, runsettings.fft_points, runsettings.time_feature_size,
        randomize=runsettings.test_randomize_data, max_samples=runsettings.test_on_n_samples,
        normalize=runsettings.normalize_data, predict_on_time_windows=runsettings.predict_on_time_windows
    )

    test_data_loader = DataLoader(test_dataset, batch_size=runsettings.test_batch_size)

    runsettings.net.to(device)
    runsettings.net.eval()

    print('Start predicting')

    test_metric_evaluator = MetricsEvaluator(
        writer, test_dataset, 'test', runsettings.filter_type, runsettings.features_type, runsettings.fs,
        compute_stoi_and_pesq=runsettings.test_compute_stoi_and_pesq,
        generate_audios=runsettings.test_generate_audios
    )

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_data_loader):
            runsettings.net.eval()

            noisy_speeches = sample_batched['noisy_speech']
            noisy_output_speeches = sample_batched['noisy_output_speech']
            clean_speeches = sample_batched['clean_speech']
            noises = sample_batched['noise']
            samples_idx = sample_batched['sample_idx']

            noisy_speeches = noisy_speeches.to(device)
            noisy_output_speeches = noisy_output_speeches.to(runsettings.device)
            clean_speeches = clean_speeches.to(device)
            noises = noises.to(device)

            outputs, _ = runsettings.net(noisy_speeches)
            if runsettings.normalize_data:
                outputs = sigmoid(outputs)

            mse_loss = criterion(outputs, clean_speeches)

            accumulated_samples_idx = test_dataset.accumulate_filtered_frames(outputs.cpu(), samples_idx)
            test_metric_evaluator.add_metrics(
                mse_loss, outputs, clean_speeches, noises, noisy_output_speeches, accumulated_samples_idx
            )

            if (i_batch + 1) % runsettings.show_metrics_every_n_batches == 0:
                test_metric_evaluator.push_metrics(i_batch + 1, 1)


if __name__ == "__main__":
    predict()
