# Running configs
import torch
from torch import nn

from constants import FilterType, FeaturesType, OptimizerType
from net import ConvDenoisingRealTimeNet

filter_type = FilterType.NOISE_FILTER

features_type = FeaturesType.REAL_TIME_TIME_FREQUENCY

optimizer_type = OptimizerType.ADAM


# Dataset config
normalize_data = True
randomize_data = True

time_feature_size = None
if features_type == FeaturesType.TIME_FREQUENCY:
    time_feature_size = 64
elif features_type == FeaturesType.REAL_TIME_TIME_FREQUENCY:
    time_feature_size = 64

predict_on_time_windows = 60

fs = 16000
windows_time_size = 16e-3
overlap_percentage = 0.5

fft_points = 2 ** 9

# Optimizer config
lr = 0.001
betas = (0.9, 0.999)
momentum = 0
decay = 0.0001

# Train config
train_batch_size = 256
train_on_n_samples = None
train_generate_audios = False
train_compute_stoi_and_pesq = False

show_metrics_every_n_batches = 50
save_model_every_n_batches = 250

# Eval config
eval_training = True
eval_every_n_batches = 250
eval_on_n_samples = 500
eval_batch_size = 128
eval_generate_audios = True
eval_compute_stoi_and_pesq = False

# More running configs
store_intermediate_outputs = False
net = ConvDenoisingRealTimeNet(
    time_feature_size, predict_on_time_windows, store_intermediate_outputs=store_intermediate_outputs
)


criterion = nn.MSELoss()

device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Test config
test_batch_size = 32
test_randomize_data = True
test_on_n_samples = None
test_generate_audios = True
test_compute_stoi_and_pesq = False
test_save_filtered_audios = True
test_push_to_tensorboard = False

# Adaptive filtering
filter_size = 16
forgetting_rate = 1
af_windows_points_size = 256
af_windows_time_size = af_windows_points_size / fs
af_batch_size = 1

