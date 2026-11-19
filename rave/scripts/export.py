import logging

logging.basicConfig(level=logging.INFO)
import torch

torch.set_grad_enabled(False)

from absl import flags, app

try:
    import rave
except ImportError:
    import sys, os
    sys.path.append(os.path.abspath('.'))
    import rave
from rave.scripting import export

FLAGS = flags.FLAGS

flags.DEFINE_string('run',
                    default=None,
                    help='Path to the run to export',
                    required=True)
flags.DEFINE_bool('streaming',
                  default=False,
                  help='Enable the model streaming mode')
flags.DEFINE_bool('use_causal_stft',
                  default=True,
                  help='Replace StreamableSTFT with CausalSTFT during streaming export')
flags.DEFINE_float(
    'fidelity',
    default=.95,
    lower_bound=.1,
    upper_bound=.999,
    help='explained variance of the PCA defining the latent size, if --latent_size is not given')

flags.DEFINE_integer(
    'latent_size',
    default=None,
    lower_bound = 0,
    help = "number of PCA dimensions of the latent space, up to the full latent size (overrides --fidelity). Requires --db_path for Wasserstein models"
)

flags.DEFINE_bool(
    'no_pca', 
    default = False,
    help = "default projection is the raw latent space (no PCA), with the full latent size of the model. "
           "Automatic for Wasserstein models without --db_path"
)

flags.DEFINE_string('name', 
                     default= None,
                     help = "custom name for the scripted model (default: run name)")
flags.DEFINE_string('out_path', 
                     default= None,
                     help = "output location of scripted model")
flags.DEFINE_string('ckpt_name',
                    default='last.ckpt',
                    help='checkpoint file to export from the run (default: last.ckpt, else the most recently written one)')
flags.DEFINE_bool('ema_weights',
                  default=False,
                  help='Use ema weights if avaiable')
flags.DEFINE_integer('sr',
                     default=None,
                     help='Optional resampling sample rate')
flags.DEFINE_string('db_path',
                    default=None,
                    help='dataset used to recompute the PCA (and fidelity) from its latents')
flags.DEFINE_enum('db_partition', 'all', ['all', 'train', 'val'],
                  help='part of the dataset used with --db_path (all = train + val)')
flags.DEFINE_bool('ica', default=False, help='add an ICA projection computed on --db_path, on --latent_size whitened PCA components (default: latent size from --fidelity)')
flags.DEFINE_bool('mapper', default=False, help='add a Mapper projection (MLP autoencoder, bottleneck of --latent_size) trained on --db_path (default: latent size from --fidelity)')
flags.DEFINE_bool('sfa', default=False, help='add a slow feature analysis projection computed on --db_path, on '
                  '--latent_size whitened PCA components (default: latent size from --fidelity)')
flags.DEFINE_bool('pls', default=False, help='add a PLS projection aligned with audio --descriptors, computed on --db_path')
flags.DEFINE_bool('cca', default=False, help='add a CCA projection aligned with audio --descriptors, computed on --db_path')
flags.DEFINE_multi_enum('descriptors', None, ['loudness', 'centroid', 'bandwidth', 'flatness', 'f0'],
                        help='--pls / --cca: audio descriptors (default: all)')
flags.DEFINE_string('pitch_mode', default='pyin', help='--pls / --cca: acids_dataset F0 tracker for the f0 descriptor '
                    '(pyin, yin, and crepe / pesto / world / praat if installed)')
flags.DEFINE_integer('max_chunks', default=None, help='maximum number of dataset chunks encoded (random subset)')
flags.DEFINE_string('device', default='cpu', help='device used to encode the dataset / train the mapper')
flags.DEFINE_integer('mapper_steps', default=5000, help='mapper training steps')
flags.DEFINE_bool('cache', default=True, help='cache the --db_path latents in rave/.dimred_cache and reload them '
                  'when valid (purged with `rave dimred purge_cache`)')
flags.DEFINE_string('prior', 
                    default=None,
                    help = "path to prior (optional)")



def main(argv):
    export(FLAGS.run,
           out_path=FLAGS.out_path,
           name=FLAGS.name,
           streaming=FLAGS.streaming,
           fidelity=FLAGS.fidelity,
           latent_size=FLAGS.latent_size,
           no_pca=FLAGS.no_pca,
           ema_weights=FLAGS.ema_weights,
           sr=FLAGS.sr,
           prior=FLAGS.prior,
           use_causal_stft=FLAGS.use_causal_stft,
           ckpt_name=FLAGS.ckpt_name,
           db_path=FLAGS.db_path,
           db_partition=FLAGS.db_partition,
           ica=FLAGS.ica,
           mapper=FLAGS.mapper,
           max_chunks=FLAGS.max_chunks,
           device=FLAGS.device,
           mapper_steps=FLAGS.mapper_steps,
           sfa=FLAGS.sfa,
           pls=FLAGS.pls,
           cca=FLAGS.cca,
           descriptors=FLAGS.descriptors,
           pitch_mode=FLAGS.pitch_mode,
           cache=FLAGS.cache)


if __name__ == "__main__":
    app.run(main)
