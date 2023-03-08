import os
import click
import torch

from torch import nn
from torch.utils.data import DataLoader

from adaptivealgorithms import rls
from adaptivefilterdatasethandler import AdaptiveFilteringDataset
from runsettings import RunSettings

criterion = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running on {}'.format(device))


@click.command()
@click.option('--input-dir', prompt='Dataset to be used while training', type=str)
def predict(input_dir):
    run_settings = RunSettings()

    dataset_path = input_dir

    print('Preparing to predict')
    test_dataset_path = os.path.join(dataset_path, 'test.csv')
    test_dataset = AdaptiveFilteringDataset(
        dataset_path, test_dataset_path, run_settings.fs, run_settings.af_windows_time_size,
        run_settings.overlap_percentage, randomize=True,
        constant_filter=False
    )
    test_data_loader = DataLoader(test_dataset, batch_size=run_settings.af_batch_size)

    print('Starting testing')

    for i_batch, sample_batched in enumerate(test_data_loader):
        noisy_speech = sample_batched['noisy_speech'].numpy()[0, :]
        correlated_noise = sample_batched['correlated_noise'].numpy()[0, :]
        samples_idx = sample_batched['sample_idx'].numpy()[0]

        noise_estimation, filtered_speech, weights_n, _, _ = rls(
            correlated_noise, noisy_speech, run_settings.filter_size, lmda=run_settings.forgetting_rate
        )
        accumulated_sample_idx = test_dataset.accumulate_filtered_speech(filtered_speech, samples_idx)
        test_dataset.write_audio(accumulated_sample_idx)


if __name__ == "__main__":
    predict()
