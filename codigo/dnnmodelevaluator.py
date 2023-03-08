import os

import torch
from torch import optim, sigmoid
from torch.utils.data import DataLoader

from runsettings import RunSettings
from constants import OptimizerType
from dnnmetrics import MetricsEvaluator
from realtimednnfilterdatasethandler import RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures


class ModelEvaluator(object):
    def __init__(
            self, tensorboard_writer, dataset_path, csv_path, num_epochs, mode, dataset_max_samples,
            batch_size, generate_audios, output_dir, is_train, storage_client, show_metrics_every_n_batches,
            testing_model_evaluator=None, models_names_iterator=None, push_metrics_to_tensor_board=None,
            push_metrics_every_x_batches=None,
            compute_pesq_and_stoi=False, test_model_every_x_batches=None, save_model_every_x_batches=None,
    ):
        self.push_metrics_every_x_batches = (
            push_metrics_every_x_batches if push_metrics_every_x_batches is not None else True
        )
        self.push_metrics_to_tensor_board = (
            push_metrics_to_tensor_board if push_metrics_to_tensor_board is not None else True
        )
        self.mode = mode
        self.csv_path = csv_path
        self.storage_client = storage_client

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.csv_path = csv_path
        self.dataset_max_samples = dataset_max_samples

        self.num_epochs = num_epochs

        self.testing_model_evaluator = testing_model_evaluator
        self.test_model_every_x_batches = test_model_every_x_batches
        self.save_model_every_x_batches = save_model_every_x_batches
        self.show_metrics_every_n_batches = show_metrics_every_n_batches
        self.models_names_iterator = models_names_iterator
        self.output_dir = output_dir
        self.is_train = is_train
        self.tensorboard_writer = tensorboard_writer
        self.generate_audios = generate_audios
        self.compute_pesq_and_stoi = compute_pesq_and_stoi

        self.run_settings = RunSettings()

        if self.run_settings.optimizer_type == OptimizerType.SGD_WITH_MOMENTUM:
            self.optimizer = optim.SGD(
                self.run_settings.net.parameters(), lr=self.run_settings.lr, momentum=self.run_settings.momentum,
                weight_decay=self.run_settings.decay
            )
        elif self.run_settings.optimizer_type == OptimizerType.ADAM:
            self.optimizer = optim.AdamW(
                self.run_settings.net.parameters(), lr=self.run_settings.lr, betas=self.run_settings.betas,
                weight_decay=self.run_settings.decay
            )
        else:
            raise RuntimeError('Unknown optimizer type {}'.format(self.run_settings.optimizer_type))

        self.dataset = None
        self.data_loader = None
        self.metric_evaluator = None
        self.init_dataset()

        print(
            f'Running model evaluator in mode {self.mode} with the following configurations:\n'
            f'\t* Epochs: {self.num_epochs}\n'
            f'\t* Max samples: {self.dataset_max_samples}\n'
            f'\t* Batch Size: {self.batch_size}\n'
            f'\t* Generate audios: {self.generate_audios}\n'
            f'\t* Is train: {self.is_train}\n'
            f'\t* Test while training: {self.testing_model_evaluator is not None}\n'
            f'\t* Test model every x batches: {self.test_model_every_x_batches}\n'
            f'\t* Save model every x batches: {self.save_model_every_x_batches}\n'
            f'\t* Show metrics every x batches: {self.show_metrics_every_n_batches}\n'
            f'\t* Must push metrics every x batches: {self.push_metrics_every_x_batches}\n'
            f'\t* Compute pesq and stoi: {self.compute_pesq_and_stoi}\n'
            f'\t* Push metrics to tensorboard: {self.push_metrics_to_tensor_board}\n'
        )

    def init_dataset(self):
        self.dataset = RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(
            self.dataset_path, self.csv_path, self.run_settings.fs, self.run_settings.windows_time_size,
            self.run_settings.overlap_percentage, self.run_settings.fft_points, self.run_settings.time_feature_size,
            self.run_settings.device, randomize=self.run_settings.randomize_data, max_samples=self.dataset_max_samples,
            normalize=self.run_settings.normalize_data,
            predict_on_time_windows=self.run_settings.predict_on_time_windows,
            batch_size=self.batch_size
        )
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size)

        self.metric_evaluator = MetricsEvaluator(
            self.tensorboard_writer, self.dataset, self.mode, self.run_settings.filter_type,
            self.run_settings.features_type,
            self.run_settings.fs, generate_audios=self.generate_audios,
            push_to_tensorboard=self.push_metrics_to_tensor_board, compute_pesq_and_stoi=self.compute_pesq_and_stoi
        )

    def evaluate(self, batches_counter):
        if self.is_train:
            self.run_settings.net.train()
            self._evaluate(batches_counter)
        else:
            with torch.no_grad():
                self.run_settings.net.eval()
                self._evaluate(batches_counter)

    def _evaluate(self, batches_counter):
        # To not being constrained by gpu ram when using large batches see:
        # * https://discuss.pytorch.org/t/multiple-forward-before-backward-call/20893/2
        # * https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822/20
        # * https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient-in-pytorch-i-e-iter-size-in-caffe-prototxt/2522/14
        for i_epoch in range(self.num_epochs):
            for i_batch, sample_batched in enumerate(self.data_loader):
                if self.push_metrics_every_x_batches:
                    batches_counter += 1

                noisy_speeches = sample_batched['noisy_speech']
                clean_speeches = sample_batched['clean_speech']
                samples_idx = sample_batched['sample_idx']

                if self.is_train:
                    self.optimizer.zero_grad()

                outputs, _ = self.run_settings.net(noisy_speeches)
                if self.run_settings.normalize_data:
                    outputs = sigmoid(outputs)

                mse_loss = self.run_settings.criterion(outputs, clean_speeches)

                if self.is_train:
                    mse_loss.backward()

                    self.optimizer.step()

                outputs_cpu = outputs.cpu().detach()
                del outputs
                outputs = outputs_cpu

                accumulated_samples_idx = self.dataset.accumulate_filtered_frames(outputs, samples_idx)
                self.metric_evaluator.add_metrics(
                    mse_loss, accumulated_samples_idx
                )

                if (self.push_metrics_every_x_batches and
                        batches_counter % self.show_metrics_every_n_batches == 0):
                    self.metric_evaluator.push_metrics(batches_counter)

                if not self.is_train and (i_batch + 1) % self.show_metrics_every_n_batches == 0:
                    print(f'[{i_batch + 1}] {self.mode}')

                if (
                        self.testing_model_evaluator and
                        batches_counter % self.test_model_every_x_batches == 0
                ):
                    self.testing_model_evaluator.evaluate(batches_counter)
                    self.run_settings.net.train()

                if self.is_train and batches_counter % self.save_model_every_x_batches == 0:
                    self.run_settings.net.current_batch_number = torch.tensor(batches_counter)
                    output_model_name = next(self.models_names_iterator)

                    if 'gs' in self.output_dir:
                        temp_file_path = os.path.join(os.getcwd(), 'tmp', output_model_name)
                        torch.save(self.run_settings.net.state_dict(), temp_file_path)
                        bucket_name = self.output_dir.split('gs://')[1]
                        bucket = self.storage_client.bucket(bucket_name)
                        blob = bucket.blob(output_model_name)
                        blob.upload_from_filename(temp_file_path)
                    else:
                        file_path = os.path.join(self.output_dir, output_model_name)
                        torch.save(self.run_settings.net.state_dict(), file_path)

                    print('Model store with name {} in {}'.format(output_model_name, self.output_dir))

        if not self.push_metrics_every_x_batches:
            self.metric_evaluator.push_metrics(batches_counter)

        if not self.is_train:
            self.init_dataset()
