import hashlib
import logging
import os
import sys
from pathlib import Path

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
flags.DEFINE_string('model', default=None, required=True, help="pretrained RAVE: run folder, checkpoint, or exported .ts (.ts: CAM prior only)")
flags.DEFINE_multi_string('config', default="prior/prior_v1.gin", help="config path")
flags.DEFINE_string('db_path', default=None, required=True, help="Preprocessed dataset path")
flags.DEFINE_string('out_path', default="runs/", help="out directory path")
flags.DEFINE_multi_string('device', default="auto", help="training device (default: auto. Can be cuda, cuda:0, ..., mps, etc.)")
flags.DEFINE_integer('batch', 8, help="batch size")
flags.DEFINE_integer('n_signal', 0, help="chunk size in audio samples (default: given by prior config)")
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
flags.DEFINE_multi_string('augment', default=[], help='augmentation configurations to use (CAM prior)')
flags.DEFINE_integer('channels', 0, help="overrides the number of training channels (0 takes the one of the model; CAM prior)")
flags.DEFINE_bool('one_batch_per_file', default=False,
                  help="balances training with one random batch per file instead of flattening (CAM prior)")
flags.DEFINE_string('ckpt_name', default='last.ckpt', help="checkpoint of the RAVE run to export (CAM prior, RAVE run given as model)")
flags.DEFINE_bool('ema_weights', default=False, help="exports the EMA weights of the RAVE run (CAM prior, RAVE run given as model)")
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


def main_variational(argv):
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

def find_config(name: str):
    """path of a gin config given to --config (searched like gin does)."""
    name = add_gin_extension(name)
    for folder in [Path('.'), rave.CUSTOM_PATH, rave.BASE_PATH, rave.BASE_PATH / 'configs']:
        if (folder / name).exists():
            return folder / name
    return None


def is_cam_config() -> bool:
    """True if the configuration (--config / --override, or the config of the run to resume) is the one of the CAM prior."""
    texts = list(FLAGS.override)
    configs = list(FLAGS.config)
    if FLAGS.resume:
        resumed = rave.core.search_for_config(FLAGS.resume)
        configs = [str(resumed)] if resumed else []
    for config in configs:
        path = find_config(config) if not os.path.exists(config) else Path(config)
        if path is not None:
            texts.append(path.read_text())
    return any('CAMPrior' in t for t in texts)


def main_cam(argv):
    from rave.prior.cam import CAMPrior, load_codec
    from rave.scripting import get_prior_class_from_config
    torch.manual_seed(FLAGS.seed)
    torch.set_float32_matmul_precision('high')

    # parse configuration
    if FLAGS.resume:
        config_file = rave.core.search_for_config(FLAGS.resume)
        if config_file is None:
            logging.error('Config file not found in %s' % FLAGS.resume)
            sys.exit(1)
        gin.parse_config_files_and_bindings([config_file], FLAGS.override)
    else:
        gin.parse_config_files_and_bindings(map(add_gin_extension, FLAGS.config), FLAGS.override)
    if FLAGS.fidelity != 0.99 or FLAGS.latent_size is not None:
        logging.warning("--fidelity / --latent_size are ignored by the CAM prior (full latent space)")

    run_name = f"{FLAGS.name}_{hashlib.md5(gin.config_str().encode()).hexdigest()[:10]}"
    if FLAGS.resume:
        run_folder = Path(rave.core.search_for_config(FLAGS.resume)).resolve().parent
        run_name = run_folder.name
        FLAGS.out_path = str(run_folder.parent)
    run_dir = os.path.join(FLAGS.out_path, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # codec: frozen RAVE (run / checkpoint are exported to run_dir/rave, non-streaming)
    codec = load_codec(FLAGS.model, out_dir=run_dir, ckpt_name=FLAGS.ckpt_name, ema_weights=FLAGS.ema_weights)
    n_channels = rave.dataset.get_training_channels(FLAGS.db_path, FLAGS.channels or codec.n_channels)
    if n_channels != codec.n_channels:
        raise ValueError(f"the model has {codec.n_channels} channel(s), but {n_channels} were requested")

    # sequence length: --n_signal (audio samples) overrides the config (latent frames)
    kwargs = {}
    if FLAGS.n_signal:
        kwargs['n_frames'] = max(FLAGS.n_signal // codec.ratio, 2)
    prior = CAMPrior(codec, **kwargs)
    FLAGS.n_signal = prior.n_signal
    logging.info(f"CAM prior: {codec.latent_size} latent dimensions, ratio {codec.ratio}, "
                 f"{prior.n_frames} frames ({prior.n_signal} samples) per sequence")

    # data (same arguments as `rave train`)
    augmentations = rave.parse_augmentations(map(rave.add_gin_extension, FLAGS.augment), sr=codec.sr)
    dataset = rave.dataset.get_dataset(FLAGS, codec.sr, augmentations=augmentations,
                                       n_channels=n_channels, one_batch_per_file=FLAGS.one_batch_per_file)
    train, val = rave.dataset.split_dataset(dataset, percent=98, training_name=run_name)
    num_workers = rave.core.get_workers(FLAGS.workers)
    train = train.get_loader(batch_size=FLAGS.batch, shuffle=True, drop_last=True,
                             num_workers=num_workers, persistent_workers=num_workers > 0)
    val = val.get_loader(batch_size=FLAGS.batch, shuffle=False, drop_last=True,
                         num_workers=num_workers, persistent_workers=num_workers > 0)

    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation", filename="best")
    last_checkpoint = rave.core.ModelCheckpoint(step_period=FLAGS.save_every)
    val_check = {
        "val_check_interval": 1 if FLAGS.smoke_test else FLAGS.val_every,
        "check_val_every_n_epoch": None,
    }
    if FLAGS.smoke_test:
        val_check['limit_train_batches'] = 1
        val_check['limit_val_batches'] = 1

    accelerator, devices = rave.core.get_training_device(FLAGS.device, allow_multi=True)
    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(FLAGS.out_path, name=run_name),
        accelerator=accelerator,
        devices=devices,
        callbacks=[validation_checkpoint, last_checkpoint],
        max_epochs=300000,
        max_steps=FLAGS.max_steps,
        gradient_clip_val=prior.grad_clip,
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )

    run = rave.core.search_for_resume_checkpoint(FLAGS.resume) if FLAGS.resume else None
    if FLAGS.resume and run is None:
        logging.error('could not find a readable checkpoint in %s' % FLAGS.resume)
        sys.exit(1)
    with open(os.path.join(run_dir, "config.gin"), "w") as config_out:
        config_out.write(gin.config_str())

    trainer.fit(prior, train, val, ckpt_path=run)


def main(argv):
    if is_cam_config():
        main_cam(argv)
    else:
        main_variational(argv)


if __name__== "__main__":
    app.run(main)