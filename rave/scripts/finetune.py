import hashlib
import math
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
ad.fragments.base.FORCE_ARRAY_RESHAPE = False

import rave
import rave.core
import rave.dataset
from rave.dataset import NORMALIZE_MODES


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_string('run', None, help='Path of model to fine-tune', required=True)
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_string('out_path',
                    default="runs/",
                    help='Output folder')
flags.DEFINE_multi_string('augment',
                           default = [],
                            help = 'augmentation configurations to use')
                         
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
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_bool('one_batch_per_file', 
                  default=False, 
                  help = "balances training with one random batch per file instead of flattening")
flags.DEFINE_string('resume',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_enum('mode', 'encoder', ['encoder', 'decoder', 'full', 'alternating'],
                  help='Fine-tuning objective and update schedule')
flags.DEFINE_multi_string('train',
                           default = ["all"],
                           help = 'specify weights to train (possible values : all, encoder, decoder, discriminator, weight regexp)')
flags.DEFINE_multi_string('freeze',
                           default = [],
                           help = 'specify weights to freeze (possible values : all, encoder, decoder, discriminator, weight regexp)')  
flags.DEFINE_float('lr_encoder', 1.e-5, help='Encoder learning rate')
flags.DEFINE_float('lr_decoder', 1.e-4, help='Decoder learning rate')
flags.DEFINE_float('lr_discriminator', 1.e-4, help='Discriminator learning rate')
flags.DEFINE_integer('encoder_start_step', 0, help='Fine-tuning batches before encoder updates')
flags.DEFINE_float('beta', None, help='Override variational encoder beta (default: source value)')
flags.DEFINE_float('regularization_weight', None,
                   help='Override latent regularization weight (default: 1)')
flags.DEFINE_integer('workers',
                     default=-1,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_boolean('update_pca', default=True, help="Update latent PCA during validation; use --noupdate_pca to retain the source PCA")
flags.DEFINE_multi_string('device', default="auto", help="training device (default: auto. Can be cuda, cuda:0, ..., mps, etc.)")
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_string('normalize',
                  default="none",
                  help='Normalizes audio input before loading. Valid options : %s'%NORMALIZE_MODES)
flags.DEFINE_list('rand_pitch',
                  default=None,
                  help='activates random pitch')
flags.DEFINE_bool('ema_weights',
                  default=False,
                  help='Use ema weights if avaiable')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')
flags.DEFINE_bool('reset_discriminator', default=False, help="resets discriminator")
flags.DEFINE_bool('smoke_test', 
                  default=False,
                  help="Run training with n_batches=1 to test the model")
flags.DEFINE_bool('allow_partial_resume', default=False, help="allow partial resuming of a checkpoint")


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


class FinetuneCallback(pl.Callback):

    def on_train_start(self, trainer, pl_module):
        if pl_module.finetune_steps >= FLAGS.max_steps:
            raise ValueError('max_steps must exceed the resumed fine-tuning step')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.finetune_steps >= FLAGS.max_steps:
            trainer.should_stop = True


def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.benchmark = True

    if FLAGS.beta is not None:
        if not math.isfinite(FLAGS.beta) or FLAGS.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        FLAGS.override.append('blocks.VariationalEncoder.beta=%r'%FLAGS.beta)

    # parse configuration
    if FLAGS.reset_discriminator: 
        remove_keys = ["discriminator.*"]
        FLAGS.train.append('discriminator')
    else:
        remove_keys = None
    try:
        model, run_path = rave.load_rave_checkpoint(FLAGS.run, ema=FLAGS.ema_weights, remove_keys=remove_keys, overrides=FLAGS.override, keep_config=True)
    except FileNotFoundError:
        model, run_path = rave.load_rave_checkpoint(FLAGS.run, ema=FLAGS.ema_weights, name=None, remove_keys=remove_keys, overrides=FLAGS.override, keep_config=True)
    if FLAGS.beta is not None and not isinstance(model.encoder, rave.blocks.VariationalEncoder):
        raise ValueError('--beta requires a variational encoder')
    # The model was instantiated inside GinEnv, so its operative bindings are
    # no longer recorded after that context exits. Save the parsed bindings.
    model_config = gin.config_str()
    RUN_NAME = rave.get_run_name(run_path)

    if FLAGS.derivative:
        #TODO replace with transform
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]
    gin.constant('SAMPLE_RATE', model.sr)
    
    # parse augmentations
    augmentations = rave.parse_augmentations(map(rave.add_gin_extension, FLAGS.augment), sr=gin.query_parameter('%SAMPLING_RATE'))
    with gin.unlock_config():
        gin.bind_parameter('dataset.get_dataset.augmentations', augmentations)

    if FLAGS.encoder_start_step < 0:
        raise ValueError('encoder_start_step must be nonnegative')
    if min(FLAGS.lr_encoder, FLAGS.lr_decoder, FLAGS.lr_discriminator) <= 0:
        raise ValueError('Fine-tuning learning rates must be positive')
    model.finetune_mode = FLAGS.mode
    model.finetune_steps = 0
    model.encoder_start_step = FLAGS.encoder_start_step
    model.finetune_lrs = (FLAGS.lr_encoder, FLAGS.lr_decoder, FLAGS.lr_discriminator)
    model.finetune_regularization = FLAGS.regularization_weight
    model.no_freeze_when_warmed_up = True
    model.encoder.no_freeze_when_warmed_up = True
    model.warmed_up = True
    model.encoder.set_warmed_up(model.warmed_up)
    model.decoder.set_warmed_up(True)

    # parse keys to train / freeze 
    training_keys = rave.core.get_finetune_keys(model, FLAGS.mode, FLAGS.train, FLAGS.freeze)
    
    if len(training_keys) == 0:
        raise ValueError('No training keys found with train=%s, freeze=%s'%(FLAGS.train, FLAGS.freeze))
    logging.info("Number of parameters fine-tuned : %d"%len(training_keys))
    with gin.unlock_config():
        gin.bind_parameter("rave.configure_optimizers.weight_list", training_keys)

    # parse datasset
    dataset = rave.dataset.get_dataset(FLAGS, 
                                       model.sr,
                                       augmentations=augmentations,
                                       n_channels=model.n_channels, 
                                       one_batch_per_file=FLAGS.one_batch_per_file)

    train, val = rave.dataset.split_dataset(dataset, percent=90, training_name=RUN_NAME)
    if len(dataset) < model.latent_size:
        train, val = dataset, dataset

    model.update_pca = FLAGS.update_pca

    num_workers = rave.core.get_workers(FLAGS.workers)
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

    os.makedirs(os.path.join(FLAGS.out_path, FLAGS.name), exist_ok=True)

    accelerator, devices = rave.core.get_training_device(FLAGS.device, allow_multi=True)
    
    callbacks = [
        FinetuneCallback(),
        rave.model.FirstCheckpointCallback(),
        validation_checkpoint,
        last_checkpoint,
    ]

    run = None
    if FLAGS.resume:
        run = rave.core.search_for_run(FLAGS.resume)
        if run is None: 
            run = rave.core.search_for_run(FLAGS.resume, None)
            if run is None:
                logging.error('could not find model with ckpt=%s. Maybe provide a more detailed path: %s'%FLAGS.resume)
                exit()

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            FLAGS.out_path,
            name=FLAGS.name,
        ),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=300000,
        max_steps=-1,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        log_every_n_steps=min(30, len(train)),
        **val_check,
    )


    # Keep each version's configuration beside its checkpoints so later runs
    # with the same name cannot change the configuration used for export.
    os.makedirs(trainer.logger.log_dir, exist_ok=True)
    with open(os.path.join(trainer.logger.log_dir, "config.gin"), "w") as config_out:
        config_out.write(model_config)

    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__": 
    app.run(main)
