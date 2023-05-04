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
import rave.core
import rave.dataset


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_multi_string('config',
                          default='v2.gin',
                          help='RAVE configuration to use')
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_string('out_path',
                    default="runs/",
                    help='Output folder')
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('save_every',
                     None,
                     help='save every n steps (default: just last)')
flags.DEFINE_integer('n_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('n_channels', None, help="number of audio channels")
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_bool('derivative',
                  default=False,
                  help='Train RAVE on the derivative of the signal')
flags.DEFINE_bool('normalize',
                  default=False,
                  help='Train RAVE on normalized signals')
flags.DEFINE_bool('rand_pitch',
                  default=False,
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
    torch.backends.cudnn.benchmark = True
    # check dataset channels
    n_channels = FLAGS.n_channels or rave.dataset.get_channels_from_dataset(FLAGS.db_path)
    gin.bind_parameter('RAVE.n_channels', n_channels)
    gin.parse_config_files_and_bindings(
        map(add_gin_extension, FLAGS.config),
        FLAGS.override,
    )
    model = rave.RAVE()

    print(model)

    if FLAGS.derivative:
        model.integrator = rave.dataset.get_derivator_integrator(model.sr)[1]

    dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                       model.sr,
                                       FLAGS.n_signal,
                                       derivative=FLAGS.derivative,
                                       normalize=FLAGS.normalize,
                                       rand_pitch=FLAGS.rand_pitch,
                                       n_channels=n_channels)
    
    train, val = rave.dataset.split_dataset(dataset, 98)
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

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            FLAGS.out_path,
            name=RUN_NAME,
        ),
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            validation_checkpoint,
            last_checkpoint,
            pl.callbacks.LearningRateMonitor(logging_interval="epoch", log_momentum=False),
            rave.model.WarmupCallback(),
            rave.model.QuantizeCallback(),
            rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
        ],
        max_epochs=100000,
        max_steps=FLAGS.max_steps,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )

    run = rave.core.search_for_run(FLAGS.ckpt)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    with open(os.path.join(FLAGS.out_path, RUN_NAME, "config.gin"), "w") as config_out:
        config_out.write(gin.operative_config_str())

    trainer.fit(model, train, val, ckpt_path=run)


if __name__ == "__main__": 
    app.run(main)
