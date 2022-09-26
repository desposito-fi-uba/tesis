from enum import Enum


class FilterType(Enum):
    NOISE_FILTER = 'NOISE_FILTER'
    SPEECH_FILTER = 'SPEECH_FILTER'


class FeaturesType(Enum):
    FREQUENCY = 'FREQUENCY'
    TIME_FREQUENCY = 'TIME_FREQUENCY'
    REAL_TIME_TIME_FREQUENCY = 'REAL_TIME_TIME_FREQUENCY'


class OptimizerType(Enum):
    ADAM = 'ADAM'
    SGD_WITH_MOMENTUM = 'SGD_WITH_MOMENTUM'


class AudioType(Enum):
    CLEAN = 'CLEAN'
    NOISY = 'NOISY'
    NOISE = 'NOISE'
    FILTERED = 'FILTERED'
    CORRELATED_NOISE = 'CORRELATED_NOISE'


class AudioNotProcessed(Exception):
    pass
