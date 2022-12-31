import itertools
import json
import os
import random
import shutil
import string
from datetime import datetime
from os import listdir
from os.path import isfile, join, splitext

import click
import torch

from google.cloud import storage
from tensorboardX import SummaryWriter

from dnnmodelevaluator import ModelEvaluator
import runsettings


@click.command()
@click.option('--epochs', default=10, required=False, prompt='Amount of train epochs', type=int)
@click.option('--output-dir', prompt='Output dir for trained models and event logs (gs prefix supported)', type=str)
@click.option('--input-dir', prompt='Dataset to be used while training', type=str)
@click.option('--experiment-name', prompt='Experiment name', type=str)
@click.option('--resume-with-model', required=False, help='Name of the model to be used to resume training', type=str)
@click.option('--use-gpu', required=False, help='GPU must be used if available', type=bool)
@click.option('--resume-from-model', required=False, help='Resume from model if available', type=bool)
@click.option('--overload-settings', required=False, help='Settings to be used', type=str)
def train(
        epochs, output_dir, input_dir, experiment_name, resume_with_model, use_gpu, resume_from_model, overload_settings
):
    if use_gpu is not None and not use_gpu:
        runsettings.device = torch.device("cpu")

    if resume_from_model is None:
        resume_from_model = True

    if not experiment_name:
        experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    if overload_settings:
        storage_client = storage.Client()
        print('Downloading settings from {}'.format(overload_settings))
        bucket_name_and_file_path = overload_settings.split('gs://')[1]
        bucket_name = bucket_name_and_file_path.split('/', 1)[0]
        file_path = bucket_name_and_file_path.split('/', 1)[1]
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        overloaded_settings_path = os.path.join(os.getcwd(), 'tmp', 'overloadedsettings.json')
        blob.download_to_filename(overloaded_settings_path)
        with open(overloaded_settings_path, 'r') as f:
            overloaded_settings = json.load(f)
            for config_key, config_value in overloaded_settings.items():
                setattr(runsettings, config_key, config_value)

    print('Running on {}'.format(torch.device(runsettings.device)))
    print('Running in mode {}'.format(runsettings.filter_type))
    print('Running with features {}'.format(runsettings.features_type))
    print('Running with optimizer {}'.format(runsettings.optimizer_type))

    experiment_id = experiment_name
    logs_path = 'logs'

    event_logs_dir = join(output_dir, logs_path, experiment_id)

    storage_client = storage.Client()
    dataset_path = input_dir
    if 'gs' in input_dir:
        print('Downloading dataset from {}'.format(input_dir))
        bucket_name_and_file_path = input_dir.split('gs://')[1]
        bucket_name = bucket_name_and_file_path.split('/', 1)[0]
        file_path = bucket_name_and_file_path.split('/', 1)[1]
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        temp_tar_dataset_path = os.path.join(os.getcwd(), 'tmp', 'dataset.tar.gz')
        blob.download_to_filename(temp_tar_dataset_path)
        tar_dataset_path = temp_tar_dataset_path
    else:
        tar_dataset_path = None
        if 'tar' in input_dir:
            tar_dataset_path = input_dir

    if tar_dataset_path:
        dataset_extracted_path = os.path.join(os.getcwd(), 'tmp', 'extracted_dataset')
        print('Extracting dataset to {}'.format(dataset_extracted_path))

        shutil.unpack_archive(tar_dataset_path, dataset_extracted_path)
        print(f'New directory tree is {[x[0] for x in os.walk(dataset_extracted_path)]}')

        dataset_path = None
        for (root, dirs, files) in os.walk(dataset_extracted_path):
            if files:
                dataset_path = root
                break

        if not dataset_path:
            raise RuntimeError('Dataset path not found')

        print(f'Selected dataset path is {dataset_path}')

    print('Looking for a model to resume from in {}'.format(output_dir))
    if 'gs' in output_dir:
        bucket_name = output_dir.split('gs://')[1]
        bucket = storage_client.bucket(bucket_name)
        chosen_blob = None
        if resume_with_model:
            print('Resuming from model {}'.format(resume_with_model))
            chosen_blob = bucket.blob(resume_with_model)
        else:
            blobs = bucket.list_blobs()
            blobs = filter(lambda blob: ('.pth' in blob.name and experiment_id in blob.name), blobs)
            sorted_blobs = list(sorted(blobs, key=lambda x: x.updated, reverse=True))
            if sorted_blobs:
                chosen_blob = sorted_blobs[0]

        if chosen_blob:
            file_name = chosen_blob.name.split('/')[-1]
            # File name could be an empty string which indicates that there is not any model to load
            if file_name:
                temp_file_path = os.path.join(os.getcwd(), 'tmp', file_name)
                chosen_blob.download_to_filename(temp_file_path)

                print('Model {} loaded'.format(file_name))
                runsettings.net.load_state_dict(torch.load(temp_file_path, map_location=runsettings.device))
    else:
        if resume_from_model:
            if not resume_with_model:
                available_models_path = [
                    join(output_dir, f)
                    for f in listdir(output_dir)
                    if isfile(join(output_dir, f)) and splitext(f)[1] == '.pth' and experiment_id in splitext(f)[0]
                ]
                latest_created_model = None
                if available_models_path:
                    latest_created_model = max(available_models_path, key=os.path.getctime)
            else:
                latest_created_model = join(output_dir, resume_with_model + '.pth')

            if latest_created_model:
                print('Model {} loaded'.format(latest_created_model))
                runsettings.net.load_state_dict(torch.load(latest_created_model))

    print('Preparing to train')
    train_csv_path = os.path.join(dataset_path, 'train.csv')

    models_names = [(
            experiment_id +
            '-' +
            ''.join(random.choices(string.ascii_letters + string.digits, k=16)) +
            '.pth'
    ) for i in range(10)]
    models_names_iterator = itertools.cycle(models_names)

    runsettings.net.to(runsettings.device)

    print('Starting training with {} epochs'.format(epochs))
    batches_counter = runsettings.net.current_batch_number.item()

    print('Output tensorboard events to {}. Resuming from step {}'.format(event_logs_dir, batches_counter))
    writer = SummaryWriter(log_dir=event_logs_dir, purge_step=batches_counter)

    validator = ModelEvaluator(
        writer, dataset_path, train_csv_path, 1, 'val', runsettings.eval_on_n_samples,
        runsettings.eval_batch_size, runsettings.eval_generate_audios, output_dir, False, storage_client,
        push_metrics_every_x_batches=False
    )

    trainer = ModelEvaluator(
        writer, dataset_path, train_csv_path, epochs, 'train', runsettings.train_on_n_samples,
        runsettings.train_batch_size, runsettings.train_generate_audios, output_dir, True, storage_client,
        validation_model_evaluator=validator, models_names_iterator=models_names_iterator,
        push_metrics_every_x_batches=True
    )
    trainer.evaluate(batches_counter)


if __name__ == "__main__":
    train()
