import importlib
import itertools
import json
import os
import random
import shutil
import string
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from os import listdir
from os.path import isfile, join, splitext
from typing import Optional

import click
import torch

import numpy as np

from google.cloud import storage
from torch import nn, optim, sigmoid
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from constants import FilterType, OptimizerType
import runsettings
from dnnmetrics import MetricsEvaluator
from realtimednnfilterdatasethandler import RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures


@click.command()
@click.option('--epochs', default=10, required=False, prompt='Amount of train epochs', type=int)
@click.option('--output-dir', prompt='Output dir for trained models and event logs (gs prefix supported)', type=str)
@click.option('--input-dir', prompt='Dataset to be used while training', type=str)
@click.option('--experiment-name', prompt='Experiment name', type=str)
@click.option('--resume-with-model', required=False, help='Name of the model to be used to resume training', type=str)
@click.option('--overload-settings', required=False, help='Settings to be used', type=str)
@click.option('--use-gpu', required=False, help='GPU must be used if available', type=bool)
@click.option('--resume-from-model', required=False, help='Resume from model if available', type=bool)
def train(epochs, output_dir, input_dir, experiment_name, resume_with_model, overload_settings, use_gpu,
          resume_from_model):
    if use_gpu is not None and not use_gpu:
        runsettings.device = torch.device("cpu")

    if resume_from_model is None:
        resume_from_model = True

    print('Running on {}'.format(runsettings.device))
    
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

    print('Running in mode {}'.format(runsettings.filter_type))
    print('Running with features {}'.format(runsettings.features_type))
    print('Running with optimizer {}'.format(runsettings.optimizer_type))

    experiment_id = experiment_name
    logs_path = 'logs'

    event_logs_dir = join(output_dir, logs_path, experiment_id)

    dataset_path = os.path.join(os.getcwd(), 'tmp', 'dataset')
    storage_client = storage.Client()
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
        print('Extracting dataset to {}'.format(dataset_path))
        shutil.unpack_archive(tar_dataset_path, dataset_path)
        dataset_path = os.path.join(dataset_path, os.listdir(dataset_path)[0])
    else:
        dataset_path = input_dir

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
    train_dataset_path = os.path.join(dataset_path, 'train.csv')
    train_dataset = RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(
        dataset_path, train_dataset_path, runsettings.fs, runsettings.windows_time_size,
        runsettings.overlap_percentage, runsettings.fft_points, runsettings.time_feature_size,
        randomize=runsettings.randomize_data, max_samples=runsettings.train_on_n_samples,
        normalize=runsettings.normalize_data, predict_on_time_windows=runsettings.predict_on_time_windows
    )

    train_data_loader = DataLoader(train_dataset, batch_size=runsettings.train_batch_size)

    if runsettings.optimizer_type == OptimizerType.SGD_WITH_MOMENTUM:
        optimizer = optim.SGD(
            runsettings.net.parameters(), lr=runsettings.lr, momentum=runsettings.momentum,
            weight_decay=runsettings.decay
        )
    elif runsettings.optimizer_type == OptimizerType.ADAM:
        optimizer = optim.Adam(
            runsettings.net.parameters(), lr=runsettings.lr, betas=runsettings.betas, weight_decay=runsettings.decay
        )
    else:
        raise RuntimeError('Unknown optimizer type {}'.format(runsettings.optimizer_type))

    models_names = [(
            experiment_id +
            '-' +
            ''.join(random.choices(string.ascii_letters + string.digits, k=16)) +
            '.pth'
        ) for i in range(10)
    ]
    models_names_iterator = itertools.cycle(models_names)

    runsettings.net.to(runsettings.device)
    runsettings.net.train()

    print('Starting training with {} epochs'.format(epochs))
    batches_counter = runsettings.net.current_batch_number.item()

    print('Output tensorboard events to {}'.format(event_logs_dir))
    writer = SummaryWriter(log_dir=event_logs_dir, purge_step=batches_counter)

    train_metric_evaluator = MetricsEvaluator(
        writer, train_dataset, 'train', runsettings.filter_type, runsettings.features_type, runsettings.fs,
        compute_stoi_and_pesq=runsettings.train_compute_stoi_and_pesq, generate_audios=runsettings.train_generate_audios
    )

    model_evaluator = ModelEvaluator(writer, dataset_path)

    for i_epoch in range(epochs):

        for i_batch, sample_batched in enumerate(train_data_loader):
            runsettings.net.train()
            batches_counter += 1

            noisy_speeches = sample_batched['noisy_speech']
            noisy_output_speeches = sample_batched['noisy_output_speech']
            clean_speeches = sample_batched['clean_speech']
            noises = sample_batched['noise']
            samples_idx = sample_batched['sample_idx']

            noisy_speeches = noisy_speeches.to(runsettings.device)

            optimizer.zero_grad()

            outputs = runsettings.net(noisy_speeches)
            if runsettings.normalize_data:
                outputs = sigmoid(outputs)

            clean_speeches = clean_speeches.to(runsettings.device)
            mse_loss = runsettings.criterion(outputs, clean_speeches)

            mse_loss.backward()

            optimizer.step()

            outputs = outputs.cpu()

            if runsettings.filter_type == FilterType.NOISE_FILTER:
                clean_speeches = clean_speeches.cpu()
            else:
                noises = noises.cpu()

            accumulated_samples_idx = train_dataset.accumulate_filtered_frames(outputs, samples_idx)
            train_metric_evaluator.add_metrics(
                mse_loss, outputs, clean_speeches, noises, noisy_output_speeches, accumulated_samples_idx
            )

            if batches_counter % runsettings.show_metrics_every_n_batches == 0:
                train_metric_evaluator.push_metrics(batches_counter, i_epoch + 1)

            if runsettings.eval_training and batches_counter % runsettings.eval_every_n_batches == 0:
                model_evaluator.eval(batches_counter, i_epoch + 1)

            if batches_counter % runsettings.save_model_every_n_batches == 0:
                runsettings.net.current_batch_number = torch.tensor(batches_counter)
                output_model_name = next(models_names_iterator)
                if 'gs' in output_dir:
                    temp_file_path = os.path.join(os.getcwd(), 'tmp', output_model_name)
                    torch.save(runsettings.net.state_dict(), temp_file_path)
                    bucket_name = output_dir.split('gs://')[1]
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(output_model_name)
                    blob.upload_from_filename(temp_file_path)
                else:
                    file_path = os.path.join(output_dir, output_model_name)
                    torch.save(runsettings.net.state_dict(), file_path)

                print('Model store with name {} in {}'.format(output_model_name, output_dir))


class ModelEvaluator(object):
    def __init__(self, writer, dataset_path):
        self.val_dataset_path = os.path.join(dataset_path, 'val.csv')
        self.dataset_path = dataset_path
        self.writer = writer

    def eval(self, batches_counter, epochs_counter):
        val_dataset = RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(
            self.dataset_path, self.val_dataset_path, runsettings.fs, runsettings.windows_time_size,
            runsettings.overlap_percentage, runsettings.fft_points, runsettings.time_feature_size,
            randomize=runsettings.randomize_data, max_samples=runsettings.eval_on_n_samples,
            normalize=runsettings.normalize_data, predict_on_time_windows=runsettings.predict_on_time_windows
        )

        data_loader = DataLoader(val_dataset, batch_size=runsettings.eval_batch_size)
        val_metric_evaluator = MetricsEvaluator(
            self.writer, val_dataset, 'val', runsettings.filter_type, runsettings.features_type, runsettings.fs,
            compute_stoi_and_pesq=runsettings.eval_compute_stoi_and_pesq,
            generate_audios=runsettings.eval_generate_audios
        )

        with torch.no_grad():
            runsettings.net.eval()
            for i_batch, sample_batched in enumerate(data_loader):
                noisy_speeches = sample_batched['noisy_speech']
                noisy_output_speeches = sample_batched['noisy_output_speech']
                clean_speeches = sample_batched['clean_speech']
                noises = sample_batched['noise']
                samples_idx = sample_batched['sample_idx']

                noisy_speeches = noisy_speeches.to(runsettings.device)

                outputs = runsettings.net(noisy_speeches)

                if runsettings.normalize_data:
                    outputs = sigmoid(outputs)

                clean_speeches = clean_speeches.to(runsettings.device)
                mse_loss = runsettings.criterion(outputs, clean_speeches)

                outputs = outputs.cpu()

                clean_speeches = clean_speeches.cpu()

                accumulated_samples_idx = val_dataset.accumulate_filtered_frames(outputs, samples_idx)
                val_metric_evaluator.add_metrics(
                    mse_loss, outputs, clean_speeches, noises, noisy_output_speeches, accumulated_samples_idx
                )

        val_metric_evaluator.push_metrics(batches_counter, epochs_counter)


if __name__ == "__main__":
    train()
