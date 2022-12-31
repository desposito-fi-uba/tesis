import os

import torch
from torch import optim, sigmoid
from torch.utils.data import DataLoader

import runsettings
from constants import OptimizerType
from dnnmetrics import MetricsEvaluator
from realtimednnfilterdatasethandler import RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures
from utils import torch_mse


class ModelEvaluator(object):
    def __init__(
            self, tensorboard_writer, dataset_path, csv_path, num_epochs, mode, dataset_max_samples,
            batch_size, generate_audios, output_dir, is_train, storage_client,
            validation_model_evaluator=None, models_names_iterator=None, push_metrics_every_x_batches=None
    ):
        self.push_metrics_every_x_batches = (
            push_metrics_every_x_batches if push_metrics_every_x_batches is not None else True
        )
        self.mode = mode
        self.csv_path = csv_path
        self.storage_client = storage_client

        self.num_epochs = num_epochs

        discard_dataset_samples_idx = (
            validation_model_evaluator.dataset.samples_order if validation_model_evaluator else None
        )

        self.dataset = RealTimeNoisySpeechDatasetWithTimeFrequencyFeatures(
            dataset_path, csv_path, runsettings.fs, runsettings.windows_time_size,
            runsettings.overlap_percentage, runsettings.fft_points, runsettings.time_feature_size,
            runsettings.device, randomize=runsettings.randomize_data, max_samples=dataset_max_samples,
            normalize=runsettings.normalize_data, predict_on_time_windows=runsettings.predict_on_time_windows,
            discard_dataset_samples_idx=discard_dataset_samples_idx
        )

        self.data_loader = DataLoader(self.dataset, batch_size=batch_size)

        if runsettings.optimizer_type == OptimizerType.SGD_WITH_MOMENTUM:
            self.optimizer = optim.SGD(
                runsettings.net.parameters(), lr=runsettings.lr, momentum=runsettings.momentum,
                weight_decay=runsettings.decay
            )
        elif runsettings.optimizer_type == OptimizerType.ADAM:
            self.optimizer = optim.AdamW(
                runsettings.net.parameters(), lr=runsettings.lr, betas=runsettings.betas, weight_decay=runsettings.decay
            )
        else:
            raise RuntimeError('Unknown optimizer type {}'.format(runsettings.optimizer_type))

        self.metric_evaluator = MetricsEvaluator(
            tensorboard_writer, self.dataset, self.mode, runsettings.filter_type, runsettings.features_type,
            runsettings.fs, generate_audios=generate_audios, push_to_tensorboard=True
        )

        self.validation_model_evaluator = validation_model_evaluator
        self.models_names_iterator = models_names_iterator
        self.output_dir = output_dir
        self.is_train = is_train

    def evaluate(self, batches_counter):
        if self.is_train:
            runsettings.net.train()
            self._evaluate(batches_counter)
        else:
            with torch.no_grad():
                runsettings.net.eval()
                self._evaluate(batches_counter)

    def _evaluate(self, batches_counter):
        for i_epoch in range(self.num_epochs):
            for i_batch, sample_batched in enumerate(self.data_loader):
                if self.push_metrics_every_x_batches:
                    batches_counter += 1

                noisy_speeches = sample_batched['noisy_speech']
                noisy_output_speeches = sample_batched['noisy_output_speech']
                clean_speeches = sample_batched['clean_speech']
                samples_idx = sample_batched['sample_idx']

                if self.is_train:
                    self.optimizer.zero_grad()

                outputs, _ = runsettings.net(noisy_speeches)
                if runsettings.normalize_data:
                    outputs = sigmoid(outputs)

                mse_loss = runsettings.criterion(outputs, clean_speeches)

                if self.is_train:
                    mse_loss.backward()

                    self.optimizer.step()

                outputs = outputs.detach().cpu()
                clean_speeches.cpu()
                noisy_speeches.cpu()

                accumulated_samples_idx = self.dataset.accumulate_filtered_frames(outputs, samples_idx)
                self.metric_evaluator.add_metrics(
                    mse_loss, accumulated_samples_idx
                )

                if (self.push_metrics_every_x_batches and
                        batches_counter % runsettings.show_metrics_every_n_batches == 0):
                    self.metric_evaluator.push_metrics(batches_counter)

                if (
                        self.is_train and
                        runsettings.eval_training and
                        batches_counter % runsettings.eval_every_n_batches == 0
                ):
                    self.validation_model_evaluator.evaluate(batches_counter)
                    runsettings.net.train()

                if self.is_train and batches_counter % runsettings.save_model_every_n_batches == 0:
                    runsettings.net.current_batch_number = torch.tensor(batches_counter)
                    output_model_name = next(self.models_names_iterator)

                    if 'gs' in self.output_dir:
                        temp_file_path = os.path.join(os.getcwd(), 'tmp', output_model_name)
                        torch.save(runsettings.net.state_dict(), temp_file_path)
                        bucket_name = self.output_dir.split('gs://')[1]
                        bucket = self.storage_client.bucket(bucket_name)
                        blob = bucket.blob(output_model_name)
                        blob.upload_from_filename(temp_file_path)
                    else:
                        file_path = os.path.join(self.output_dir, output_model_name)
                        torch.save(runsettings.net.state_dict(), file_path)

                    print('Model store with name {} in {}'.format(output_model_name, self.output_dir))

        if not self.push_metrics_every_x_batches:
            self.metric_evaluator.push_metrics(batches_counter)
