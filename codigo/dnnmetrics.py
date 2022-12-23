from statistics import mean


class MetricsEvaluator(object):
    def __init__(
            self, tensorboard_writer, dataset, mode, filter_type, features_type, fs, generate_audios=True,
            push_to_tensorboard=True
    ):
        self.accumulative_mse_loss = []
        self.accumulative_max_mse_loss = []
        self.accumulative_samples_idx = []
        self.writer = tensorboard_writer
        self.dataset = dataset
        self.mode = mode

        self.filter_type = filter_type
        self.features_type = features_type
        self.fs = fs

        self.generate_audios = generate_audios
        self.push_to_tensorboard = push_to_tensorboard

    def restart_metrics(self):
        self.accumulative_mse_loss = []
        self.accumulative_max_mse_loss = []
        self.accumulative_samples_idx = []

    def add_metrics(self, mse_loss, max_mse_loss, samples_idx):

        self.accumulative_mse_loss.append(mse_loss.item())
        self.accumulative_max_mse_loss.append(max_mse_loss.item())

        if not self.generate_audios:
            return

        for sample_idx in samples_idx:
            self.dataset.write_audio(sample_idx)

        self.accumulative_samples_idx.extend(samples_idx)

    def push_metrics(self, batches_counter):
        mse_loss_value = mean(self.accumulative_mse_loss)
        max_mse_loss_value = mean(self.accumulative_max_mse_loss)

        print(
            '[{}] {}, '
            'mse_loss: {:.5e}, '
            'max_mse_loss: {:.5f}'.format(
                batches_counter, self.mode, mse_loss_value, max_mse_loss_value
            )
        )

        if self.push_to_tensorboard:
            self.writer.add_scalars('MSELoss/{}'.format(self.mode), {
                'max': max_mse_loss_value,
                'actual': mse_loss_value
            }, batches_counter)
            self.writer.flush()

        self.restart_metrics()
