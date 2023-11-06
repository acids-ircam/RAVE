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

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run')
flags.DEFINE_string('model', default=None, required=True, help="pretrained RAVE path")
flags.DEFINE_multi_string('config', default="prior/prior_v1.gin", help="config path")
flags.DEFINE_string('db_path', default=None, required=True, help="Preprocessed dataset path")
flags.DEFINE_string('out_path', default="runs/", help="out directory path")
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_integer('batch', 8, help="batch size")
flags.DEFINE_integer('n_signal', 0, help="chunk size (default: given by prior config)")
flags.DEFINE_string('ckpt', default=None, help="checkpoint to resume")
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('save_every',
                     None,
                     help='save every n steps (default: just last)')
flags.DEFINE_integer('max_steps', default=1000000, help="max training steps")
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')

flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_bool('normalize',
                  default=False,
                  help='Train RAVE on normalized signals')
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

    # load pretrained RAVE
    config_file = rave.core.search_for_config(FLAGS.model) 
    if config_file is None:
        print('no configuration file found at address :'%FLAGS.model)
    gin.parse_config_file(config_file)
    run = rave.core.search_for_run(FLAGS.model)
    if run is None:
        print('no checkpoint found in %s'%FLAGS.model)
        exit()
    pretrained = rave.RAVE()
    print('model found : %s'%run)
    checkpoint = torch.load(run, map_location='cpu')
    if "EMA" in checkpoint["callbacks"]:
        pretrained.load_state_dict(
            checkpoint["callbacks"]["EMA"],
            strict=False,
        )
    else:
        pretrained.load_state_dict(
            checkpoint["state_dict"],
            strict=False,
        )
    pretrained.eval()
    gin.clear_config()
    
    # parse configuration
    if FLAGS.ckpt:
        config_file = rave.core.search_for_config(FLAGS.ckpt)
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
        prior = rave.prior.VariationalPrior(pretrained_vae=pretrained)
    else:
        raise NotImplementedError("prior not implemented for encoder of type %s"%(type(pretrained.encoder)))

    dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                       pretrained.sr,
                                       max(FLAGS.n_signal, prior.min_receptive_field),
                                       derivative=FLAGS.derivative,
                                       normalize=FLAGS.normalize,
                                       rand_pitch=FLAGS.rand_pitch,
                                       n_channels=pretrained.n_channels)

    train, val = rave.dataset.split_dataset(dataset, 98)

    # get data-loader
    num_workers = FLAGS.workers
    if os.name == "nt" or sys.platform == "darwin":
        num_workers = 0
    train = DataLoader(train,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, FLAGS.batch, False, num_workers=num_workers)

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

    gin_hash = hashlib.md5(
        gin.operative_config_str().encode()).hexdigest()[:10]

    RUN_NAME = f'{FLAGS.name}_{gin_hash}'
    os.makedirs(os.path.join(FLAGS.out_path, RUN_NAME), exist_ok=True)

    if FLAGS.gpu == [-1]:
        gpu = 0
    else:
        gpu = FLAGS.gpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    if FLAGS.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = FLAGS.gpu or rave.core.setup_gpu()
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

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

    run = rave.core.search_for_run(FLAGS.ckpt)
    if run is not None:
        print('loading state from file %s'%run)
        loaded = torch.load(run, map_location='cpu')
        trainer.fit_loop.epoch_loop._batches_that_stepped = loaded['global_step']
    
    with open(os.path.join(FLAGS.out_path, RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    trainer.fit(prior, train, val, ckpt_path=run)

if __name__== "__main__":
    app.run(main)