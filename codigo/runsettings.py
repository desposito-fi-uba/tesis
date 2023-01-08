from abc import ABCMeta

from torch import nn

from constants import FilterType, FeaturesType, OptimizerType
from net import ConvDenoisingRealTimeNet


class Singleton(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RunSettings(metaclass=Singleton):

    def __init__(self):
        self.fs = 16000
        self.windows_time_size = 16e-3
        self.overlap_percentage = 0.5

        # self.fft_points = 2 ** 10
        self.fft_points = 2 ** 9
        self.freq_feature_size = self.fft_points // 2

        if self.fft_points < self.fs * self.windows_time_size:
            raise RuntimeError(
                f'N {self.fft_points} must be >= than L {self.fs * self.windows_time_size}, '
                f'to prevent aliasing in time domain'
            )

        # Dnn filter
        self.filter_type = FilterType.NOISE_FILTER
        self.features_type = FeaturesType.REAL_TIME_TIME_FREQUENCY
        self.optimizer_type = OptimizerType.ADAM

        self.normalize_data = True
        self.randomize_data = True

        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.momentum = 0
        self.decay = 0.001

        self.time_feature_size = 64

        self.predict_on_time_windows = 60

        self.train_batch_size = 400
        self.train_on_n_samples = None
        self.train_generate_audios = False
        self.train_compute_stoi_and_pesq = False
        self.test_while_training = True
        self.test_while_training_every_n_batches = 1000

        self.show_metrics_every_n_batches = 50
        self.save_model_every_n_batches = 1000

        self.test_batch_size = 400
        self.test_randomize_data = True
        self.test_on_n_samples = 500
        self.test_generate_audios = False
        self.test_compute_stoi_and_pesq = True
        self.test_save_filtered_audios = False

        self.store_intermediate_outputs = False
        self.net = ConvDenoisingRealTimeNet(
            self.time_feature_size, self.freq_feature_size, self.predict_on_time_windows,
            store_intermediate_outputs=self.store_intermediate_outputs
        )

        self.criterion = nn.MSELoss()
        self.device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Adaptive filtering
        self.filter_size = 16
        self.forgetting_rate = 1
        self.af_windows_points_size = 256
        self.af_windows_time_size = self.af_windows_points_size / self.fs
        self.af_batch_size = 1

