import hashlib
import os
import sys

import gin
import pytorch_lightning as pl
import torch
from absl import flags, app
from torch.utils.data import DataLoader

try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave

import rave
import rave.dataset
import rave.prior

from rave.dataset import NORMALIZE_MODES

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run')
flags.DEFINE_string('model', default=None, required=True, help="pretrained RAVE path")
flags.DEFINE_multi_string('config', default="prior/prior_v1.gin", help="config path")
flags.DEFINE_string('db_path', default=None, required=True, help="Preprocessed dataset path")
flags.DEFINE_string('out_path', default="runs/", help="out directory path")
flags.DEFINE_multi_string('device', default="auto", help="training device (default: auto. Can be cuda, cuda:0, ..., mps, etc.)")
flags.DEFINE_integer('batch', 8, help="batch size")
flags.DEFINE_integer('n_signal', 0, help="chunk size (default: given by prior config)")
flags.DEFINE_string('resume', default=None, help="path of checkpoint to resume")
flags.DEFINE_float('fidelity', default=0.99, help="prior target fidelity (exclusive with fidelity)")
flags.DEFINE_integer('latent_size', default=None, help="enforces learned latent size (exclusive with fidelity)")
flags.DEFINE_integer('workers',
                     default=None,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('save_every',
                     200000,
                     help='Save a numbered checkpoint every n training batches (default: 200000)')
flags.DEFINE_integer('max_steps', default=1000000, help="max training steps")
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('seed',
                    default = 0,
                    help = 'augmentation configurations to use')  
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_string('normalize',
                  default="none",
                  help='Normalizes audio input before loading. Valid options : %s'%NORMALIZE_MODES)
flags.DEFINE_list('rand_pitch',
                  default=None,
                  help='activates random pitch')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')
flags.DEFINE_bool('smoke_test', 
                  default=False,
                  help="Run training with n_batches=1 to test the model")


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(argv):
    torch.manual_seed(FLAGS.seed)
    # load pretrained RAVE
    try:
        pretrained, run_path = rave.load_rave_checkpoint(FLAGS.model)
    except FileNotFoundError:
        pretrained, run_path = rave.load_rave_checkpoint(FLAGS.model, name=None)
    n_channels = getattr(pretrained, "n_channels", 1)
    gin.clear_config()
    
    # parse configuration
    gin.constant('SAMPLE_RATE', pretrained.sr)
    if FLAGS.resume:
        config_file = rave.core.search_for_config(FLAGS.resume)
        if config_file is None:
            print('Config gile not found in %s'%FLAGS.run)
        gin.parse_config_file(config_file)
    else:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, FLAGS.config),
            FLAGS.override
        )

    # create model
    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        prior = rave.prior.VariationalPrior(pretrained_vae=pretrained, fidelity=FLAGS.fidelity, latent_size=FLAGS.latent_size, n_channels=n_channels)
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        prior = rave.prior.WassersteinPrior(pretrained_vae=pretrained, fidelity=FLAGS.fidelity, latent_size=FLAGS.latent_size, n_channels=n_channels)
    else:
        raise NotImplementedError("prior not implemented for encoder of type %s"%(type(pretrained.encoder)))

    FLAGS.n_signal = max(FLAGS.n_signal, prior.min_receptive_field)
    dataset = rave.dataset.get_dataset(FLAGS,
                                       pretrained.sr,
                                       n_channels=pretrained.n_channels, 
                                       one_batch_per_file=False
                                       )

    train, val = rave.dataset.split_dataset(dataset, percent=98, training_name=rave.get_run_name(run_path))

    # get data-loader
    num_workers = rave.core.get_workers(FLAGS.workers)
    train = train.get_loader(
        FLAGS.batch, True, drop_last=True, num_workers=num_workers, persistent_workers=True
    )
    val = val.get_loader(
        FLAGS.batch, False, num_workers=num_workers, persistent_workers=True
    )

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best")
    last_checkpoint = rave.core.ModelCheckpoint(step_period=FLAGS.save_every)

    val_check = {}
    if len(train) >= FLAGS.val_every:
        val_check["val_check_interval"] = 1 if FLAGS.smoke_test else FLAGS.val_every
    else:
        nepoch = FLAGS.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    if FLAGS.smoke_test:
        val_check['limit_train_batches'] = 1
        val_check['limit_val_batches'] = 1

    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]

    

    RUN_NAME = f'{FLAGS.name}_{gin_hash}'
    os.makedirs(os.path.join(FLAGS.out_path, RUN_NAME), exist_ok=True)

    accelerator, devices = rave.core.get_training_device(FLAGS.device, allow_multi=True)

    callbacks = [
        validation_checkpoint,
        last_checkpoint,
    ]

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
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )

    run = rave.core.search_for_run(FLAGS.resume)
    if run is not None:
        print('loading state from file %s'%run)
        loaded = torch.load(run, map_location='cpu')
        trainer.fit_loop.epoch_loop._batches_that_stepped = loaded['global_step']
    
    with open(os.path.join(FLAGS.out_path, RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    trainer.fit(prior, train, val, ckpt_path=run)

if __name__== "__main__":
    app.run(main)