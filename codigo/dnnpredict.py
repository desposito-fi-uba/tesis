import os
from os.path import isfile, join

import click
import torch
import numpy as np

from tensorboardX import SummaryWriter

from dnnmodelevaluator import ModelEvaluator
import runsettings


@click.command()
@click.option('--model', required=False, help='Name of the model to be used to predict', type=str)
@click.option('--output-dir', prompt='Output dir for trained models and event logs (gs prefix supported)', type=str)
@click.option('--input-dir', prompt='Dataset to be used while training', type=str)
@click.option('--experiment-name', prompt='Experiment name', type=str)
def predict(output_dir, input_dir, experiment_name, model=None):
    print('Running on {}'.format(torch.device(runsettings.device)))
    print('Running in mode {}'.format(runsettings.filter_type))
    print('Running with features {}'.format(runsettings.features_type))
    print('Running with optimizer {}'.format(runsettings.optimizer_type))

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

    tester = ModelEvaluator(
        writer, dataset_path, test_dataset_path, 1, 'test', runsettings.test_on_n_samples,
        runsettings.test_batch_size, runsettings.test_generate_audios, output_dir, False, None,
        push_metrics_every_x_batches=True
    )
    tester.evaluate(0)


if __name__ == "__main__":
    predict()
