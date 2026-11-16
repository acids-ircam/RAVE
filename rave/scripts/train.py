import hashlib
import pdb
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Dict

import gin
import pytorch_lightning as pl
import torch
from absl import flags, app, logging
from torch.utils.data import DataLoader

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave

import acids_dataset as ad
ad.fragments.base.FORCE_ARRAY_RESHAPE = False #TODO remove?

import rave
import rave.core
import rave.dataset
from rave.dataset import NORMALIZE_MODES


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_multi_string('config',
                          default='v2.gin',
                          help='RAVE configuration to use')
flags.DEFINE_multi_string('augment',
                           default = [],
                            help = 'augmentation configurations to use')
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_string('out_path',
                    default="runs/",
                    help='Output folder')
flags.DEFINE_float('beta', 
                   default = None, 
                   help = "Set manually regularization term (variational / wasserstein only)")
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('save_every',
                     500000,
                     help='save every n steps (default: just last)')
flags.DEFINE_integer('seed',
                           default = 0,
                           help = 'augmentation configurations to use')                    
flags.DEFINE_integer('n_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_bool('stereo', False, help="flag to force stereo training")
flags.DEFINE_bool('midside', False, help="flag to force stereo mid-side training")
flags.DEFINE_bool('ambisonics', False, help="activates ambisonic mode : encodes the input signal before training")
flags.DEFINE_integer('channels', 0, help="overrides the exact number of training channels (0 takes the one from dataset)")
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('resume',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_multi_string('init_from', 
                          [], 
                          help="Transfer keys from an existing model. Must be of format path_to_ckpt=>pattern1,pattern2 ; see README.md for more explanation")
flags.DEFINE_bool('strict_init', default=True, help = "allows uncomplete transfer if keys are not found (default: None)")                        
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=-1,
                     help='Number of workers to spawn for dataset loading (default: 0, set -1 for automatic mode)')
flags.DEFINE_multi_string('device', default="auto", help="training device (default: auto. Can be cuda, cuda:0, ..., mps, etc.)")
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_string('normalize',
                  default="none",
                  help='Normalizes audio input before loading. Valid options : %s'%NORMALIZE_MODES)
flags.DEFINE_bool('one_batch_per_file', 
                  default=False, 
                  help = "balances training with one random batch per file instead of flattening")
flags.DEFINE_bool('normalize_dataset', 
                  default=False, 
                  help="Pre-computes loundess of the whole dataset and normalizes input / output data.")
flags.DEFINE_list('rand_pitch',
                  default=None,
                  help='activates random pitch')
flags.DEFINE_float('ema',
                   default=None,
                   help='Exponential weight averaging factor (optional)')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')
flags.DEFINE_bool('smoke_test', 
                  default=False,
                  help="Run training with n_batches=1 to test the model")
flags.DEFINE_bool('allow_partial_resume', default=False, help="allow partial resuming of a checkpoint")


class EMA(pl.Callback):

    def __init__(self, factor=.999) -> None:
        super().__init__()
        self.weights = {}
        self.factor = factor

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx) -> None:
        for n, p in pl_module.named_parameters():
            if n not in self.weights:
                self.weights[n] = p.data.clone()
                continue

            self.weights[n] = self.weights[n] * self.factor + p.data * (
                1 - self.factor)

    def swap_weights(self, module):
        for n, p in module.named_parameters():
            current = p.data.clone()
            p.data.copy_(self.weights[n])
            self.weights[n] = current

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.weights:
            self.swap_weights(pl_module)
        else:
            print("no ema weights available")

    def state_dict(self) -> Dict[str, Any]:
        return self.weights.copy()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.weights.update(state_dict)

def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True

    # check dataset channels
    n_channels = rave.dataset.get_training_channels(FLAGS.db_path, FLAGS.channels, FLAGS.stereo, FLAGS.midside, FLAGS.ambisonics) 
    FLAGS.override.append('RAVE.n_channels=%d'%n_channels)
    if FLAGS.beta: FLAGS.override.append("model.BetaWarmupCallback.target_value=%r"%FLAGS.beta)
    
    # parse configuration
    if FLAGS.resume:
        config_file = rave.core.search_for_config(FLAGS.resume)
        if config_file is None:
            logging.error('Config file not found in %s'%FLAGS.run)
            exit()
        FLAGS.config = list(filter(lambda x: x in rave.RAVE_OVERRIDE_CONFIGS, map(rave.add_gin_extension, FLAGS.config)))
        gin.parse_config_files_and_bindings([config_file] + FLAGS.config, FLAGS.override)
    else:
        gin.parse_config_files_and_bindings(
            map(rave.add_gin_extension, FLAGS.config),
            FLAGS.override,
        )

    # parse augmentations
    augmentations = rave.parse_augmentations(map(rave.add_gin_extension, FLAGS.augment), sr=gin.query_parameter('%SAMPLING_RATE'))
    with gin.unlock_config():
        gin.bind_parameter('dataset.get_dataset.augmentations', augmentations)


    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]
    RUN_NAME = f'{FLAGS.name}_{gin_hash}'

    # parse datasset
    dataset = rave.dataset.get_dataset(
                                       FLAGS, 
                                       gin.query_parameter('%SAMPLING_RATE'),
                                       augmentations=augmentations,
                                       n_channels=n_channels, 
                                       one_batch_per_file=FLAGS.one_batch_per_file
                                       )

    train, val = rave.dataset.split_dataset(dataset, percent=90, training_name=RUN_NAME)


    # create model
    model = rave.RAVE() # type: ignore

    # transfer keys if needed
    if len(FLAGS.init_from) > 0: 
        for pattern in FLAGS.init_from:
            weight_dict = rave.core.get_weights_from_pattern(pattern)
            model.import_weights(weight_dict, strict=FLAGS.strict_init)

    if FLAGS.derivative:
        #TODO replace with transform
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    

    num_workers = rave.core.get_workers(FLAGS.workers)
    # train = DataLoader(train,
    #                    FLAGS.batch,
    #                    True,
    #                    drop_last=True,
    #                    num_workers=num_workers)
    # val = DataLoader(val, FLAGS.batch, False, num_workers=num_workers)
    train = train.get_loader(
        batch_size=FLAGS.batch,
        shuffle=True, 
        drop_last=True, 
        num_workers = num_workers, 
        persistent_workers=True

    )
    val = val.get_loader(
        batch_size=FLAGS.batch,
        shuffle=False, 
        drop_last=True, 
        num_workers = num_workers, 
        persistent_workers=True

    )

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best")
    last_filename = "last" if FLAGS.save_every is None else "epoch-{epoch:04d}"                                                        
    last_checkpoint = rave.core.ModelCheckpoint(filename=last_filename, step_period=FLAGS.save_every)

    val_check = {}
    if len(train) >= FLAGS.val_every:
        val_check["val_check_interval"] = 1 if FLAGS.smoke_test else FLAGS.val_every
    else:
        nepoch = FLAGS.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    if FLAGS.smoke_test:
        val_check['limit_train_batches'] = 1
        val_check['limit_val_batches'] = 1

    os.makedirs(os.path.join(FLAGS.out_path, RUN_NAME), exist_ok=True)

    accelerator, devices = rave.core.get_training_device(FLAGS.device, allow_multi=True)
    
    callbacks = [
        rave.model.FirstCheckpointCallback(),
        validation_checkpoint,
        last_checkpoint,
        rave.model.WarmupCallback(),
        rave.model.QuantizeCallback(),
        rave.model.BetaWarmupCallback(),
    ]

    if FLAGS.ema is not None:
        callbacks.append(EMA(FLAGS.ema))

    run = None
    if FLAGS.resume:
        run = rave.core.search_for_run(FLAGS.resume)
        if run is None: 
            run = rave.core.search_for_run(FLAGS.resume, None)
            if run is None:
                logging.error('could not find model with ckpt=%s. Maybe provide a more detailed path'%FLAGS.resume)
                exit()

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            FLAGS.out_path,
            name=RUN_NAME,
        ),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=300000,
        max_steps=FLAGS.max_steps,
        profiler="simple",
        # limit_train_batches=1,
        # limit_val_batches=30,
        # fast_dev_run=True,
        enable_progress_bar=FLAGS.progress,
        log_every_n_steps=min(30, len(train)),
        **val_check,
    )


    with open(os.path.join(FLAGS.out_path, RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.config_str())

    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__": 
    app.run(main)
