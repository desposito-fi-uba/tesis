import torch
from torch import nn


class ConvDenoisingRealTimeNet(nn.Module):

    def __init__(self, time_feature_size, predict_on_time_windows, store_intermediate_outputs=None):
        super().__init__()

        self.store_intermediate_outputs = store_intermediate_outputs

        down = [(2 * 512, 256), (2 * 256, 128), (2 * 128, 64), (2 * 64, 32), (2 * 32, 16)]
        up = [(16, 32), (32, 64), (64, 128), (128, 256), (256, 512)]
        middle_block = 512
        first_last_layer = 16

        self.input_layer = self.build_input_layer(1, first_last_layer)
        encoder_layers = [
            self.build_block(in_c, out_c) for in_c, out_c
            in up
        ]
        self.encoder = nn.ModuleList(encoder_layers)
        self.encoder_pool_layers = nn.ModuleList(
            [nn.MaxPool2d(2) for i in range(0, len(self.encoder))]
        )

        self.middle_layer = self.build_block(middle_block, middle_block)

        decoder_layers = [
            self.build_block(in_c, out_c) for in_c, out_c
            in down
        ]
        self.decoder = nn.ModuleList(decoder_layers)
        self.decoder_up_sampling_layers = nn.ModuleList(
            [nn.UpsamplingBilinear2d(scale_factor=2) for i in range(0, len(self.encoder))]
        )

        self.output_layer = self.build_output_layer(first_last_layer, 1)

        self.register_buffer("current_batch_number", torch.tensor(0))

        self.time_feature_size = time_feature_size
        self.predict_on_time_windows = predict_on_time_windows

    def forward(self, x):
        intermediate_outputs = []

        x = self.input_layer(x)

        if self.store_intermediate_outputs:
            intermediate_outputs.append(x)

        connections = []
        for encoder_layer, pool_layer in zip(self.encoder, self.encoder_pool_layers):
            x = encoder_layer(x)
            connections.append(x)
            x = pool_layer(x)

            if self.store_intermediate_outputs:
                intermediate_outputs.append(x)

        x = self.middle_layer(x)

        if self.store_intermediate_outputs:
            intermediate_outputs.append(x)

        for decoded_layer, up_sampling_layer in zip(self.decoder, self.decoder_up_sampling_layers):
            x = up_sampling_layer(x)
            x = torch.cat((x, connections.pop()), dim=1)
            x = decoded_layer(x)

            if self.store_intermediate_outputs:
                intermediate_outputs.append(x)

        x = self.output_layer(x)

        end_index = self.time_feature_size - (2 * (self.time_feature_size - self.predict_on_time_windows))
        start_index = end_index - 3
        x = x[:, :, :, start_index:end_index]

        return x, intermediate_outputs

    def build_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

    def build_input_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

    def build_output_layer(self, in_channels, out_channels):
        layers = [
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ]
        return nn.Sequential(*layers)
