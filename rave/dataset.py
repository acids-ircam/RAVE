import torch
from absl import logging
from typing import Iterable, Tuple, List
import math
import numpy as np
import contextlib
from pathlib import Path
import os, yaml
import gin.torch
import acids_dataset as ad
from scipy.signal import lfilter

NORMALIZE_MODES = ['none', 'dataset', 'dataset_compress', 'each', 'each_compress']

@gin.configurable(module="data")
def split_dataset(dataset, 
                  partition: str | None = None, 
                  percent: float = 95, 
                  features: str | List[str] | None = None, 
                  balance_cardinality: bool = False, 
                  max_residual: int | None = None,
                  training_name: str | None = None) -> Tuple[ad.datasets.AudioDataset, ad.datasets.AudioDataset]:
    force = os.environ.get('RAVE_FORCE_PARTITION', '') not in ('', '0')
    if partition is not None and not force:
        assert percent is None, "either percent or partition keyword must be given."
        partdict = dataset.load_partition(partition)
    elif (training_name is not None) and (dataset.has_partition(training_name)) and (not force): 
        partdict = dataset.load_partition(training_name)
    else: 
        if force: 
            logging.info('forcing new partitions in dataset')
        partdict = dataset.split(partitions={'train': percent / 100, 'test': 1 - percent / 100},
                                 features=features, 
                                 balance_cardinality=balance_cardinality,
                                 write=training_name, 
                                 max_items={'test': 1000},
                                 by="auto") 
    # persistant random shuffling for validation dataset
    partdict['test'].shuffle_idx = 1010
    return partdict['train'], partdict['test']


@gin.configurable
def get_output_transform(transform):
    return transform

def parse_output_transform(gin_config):
    with gin.unlock_config():
        gin.parse_config_file(str(Path("output_transforms") / Path(gin_config).with_suffix('.gin')))
    return get_output_transform() #type: ignore


@gin.configurable
def get_dataset(flags,
                sr,
                augmentations: Iterable[ad.transforms.Transform] | None = None, 
                n_channels: int | None = None, 
                one_batch_per_file: bool = False): 
                # derivative: bool = False,
                # normalize: str | None = None,
                # rand_pitch: Tuple[float, float] | None = None,
                # n_channels: int | None = None, **kwargs):
    n_signal = flags.n_signal
    assert n_channels is not None
    dataset = ad.datasets.AudioDataset(
        flags.db_path,
        sr=sr,
        channels=n_channels, 
        chunk_length = flags.n_signal, 
        batch_by="files" if one_batch_per_file else "dataset", 
        float_dtype=torch.float32

    )
    
    assert flags.normalize in NORMALIZE_MODES
    normalize_transform = None
    if flags.normalize == "each": 
        normalize_transform = ad.transforms.Normalize(sr=sr, p=1.)
    elif flags.normalize == "compress": 
        normalize_transform = ad.transforms.CompressRange((-40, -3))
    elif flags.normalize == "dataset": 
        dataset_peak = dataset.sample_features(ad.features.Peak(sr=sr), reduction=max)
        normalize_transform = ad.transforms.NormRange(dataset_peak)
    elif flags.normalize == "dataset_compress":
        dataset_range = dataset.sample_features(ad.features.LoudnessMeter("db_quantiles", sr=sr), reduction=max)
        normalize_transform = ad.transforms.CompressRange((-40, -3), dataset_range, mode="db")

    transform_list = []
    output_transform = None
    # stretch or crop
    if flags.rand_pitch:
        transform_list.append(ad.transforms.Stretch(n_signal, flags.rand_pitch), sr=sr)
    else:
        transform_list.append(ad.transforms.Crop(n_signal, sr=sr))
        
    transform_list.extend([
        ad.transforms.Dequantize(16, sr=sr),
        ad.transforms.PhaseMangle(sr=sr),
    ])

    if flags.normalize:
        transform_list.append(normalize_transform)

    if flags.derivative:
        transform_list.append(ad.transforms.Derivator(sr=sr))

    if augmentations:
        transform_list.extend(augmentations)

    if getattr(flags, "midside", False): 
        output_transform = parse_output_transform('midside')
        transform_list.append(output_transform)
    elif getattr(flags, "ambisonics", False):
        with gin.unlock_config(): 
            gin.constant('AMBISONICS_ORDER', int(math.sqrt(n_channels - 1)))
        output_transform = parse_output_transform('ambisonics')
        transform_list.append(output_transform)
    dataset.transforms = transform_list
    return dataset

def get_channels_from_dataset(db_path):
    if ad.is_preprocessed(db_path): 
        with open(os.path.join(db_path, 'metadata.yaml'), 'r') as metadata:
            metadata = yaml.safe_load(metadata)
        return metadata.get('channels')
    else:
        return None

def get_training_channels(db_path, target_channels, stereo=False, mid_side=False, ambisonics=False):
    dataset_channels = get_channels_from_dataset(db_path)
    assert target_channels>=0, "invalid number of channels : %s"%target_channels
    assert sum([target_channels != 0, stereo, mid_side, ambisonics]) < 2, "provided several values for channels; please only give one"
    if stereo or mid_side: 
        target_channels = 2
    elif ambisonics: 
        if dataset_channels is None: dataset_channels = ad.get_channels_from_path(db_path, write=True)
        if len(dataset_channels) > 1: raise ValueError('dataset has disparate number of channels. Please ensure all files are matching a correct ambisonic order (order+1)^2')
        dataset_channels = list(dataset_channels)[0]
        if dataset_channels < 4: raise ValueError('dataset seems to have 1 channel; must be at least 4')
        order = math.sqrt(dataset_channels)
        if not order.is_integer():
            raise ValueError('dataset seems to be %d channels, that is invalid with ambisonics encoding')
        target_channels = dataset_channels
    elif target_channels == 0: 
        target_channels = dataset_channels or max(ad.get_channels_from_path(db_path, write=True))

    if dataset_channels is not None:
        if target_channels > dataset_channels:
            raise RuntimeError('[Error] Requested number of channels is %s, but dataset has %s channels'%(target_channels, dataset_channels))
    n_channels = target_channels
    if n_channels is None:
        n_channels = None
    return n_channels


# Utilitary for GIN recording of augmentations

_augmentations = []

@gin.configurable(module="transforms")
def parse_transform(transform):
    global _augmentations
    if isinstance(transform, type): 
        transform = transform()
    _augmentations.append(transform)

def get_augmentations():
    return _augmentations

def get_derivator_integrator(sr: int):
    alpha = 1 / (1 + 1 / sr * 2 * np.pi * 10)
    derivator = ([.5, -.5], [1])
    integrator = ([alpha**2, -alpha**2], [1, -2 * alpha, alpha**2])
    return lambda x: lfilter(*derivator, x), lambda x: lfilter(*integrator, x)

