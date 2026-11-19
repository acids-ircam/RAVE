
import sys; sys.path.append('../acids-dataset')
import acids_dataset

import torch
from absl import app, flags

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path',
                          None,
                          help='Path to a directory containing audio files',
                          required=True)
flags.DEFINE_string('output_path',
                    None,
                    help='Output directory for the dataset',
                    required=True)
flags.DEFINE_integer('num_signal',
                     131072,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('channels', 1, help="Number of audio channels")
flags.DEFINE_integer('sampling_rate',
                     44100,
                     help='Sampling rate to use during training')
flags.DEFINE_integer('max_db_size',
                     100,
                     help='Maximum size (in GB) of the dataset')

flags.DEFINE_multi_string('filter', [], help="wildcard to filter target files")
flags.DEFINE_multi_string('exclude', [], help="wildcard to exclude target files")
flags.DEFINE_multi_string('meta_regexp', [], help="additional regexp for metadata parsing")
flags.DEFINE_multi_string('feature', [], help="additional feature files (see acids-dataset)")



def main(args):
    acids_dataset.preprocess_dataset(
        FLAGS.input_path, 
        FLAGS.output_path, 
        config = "rave.gin",
        features = FLAGS.feature,
        chunk_length = 2 * FLAGS.num_signal, 
        sample_rate = FLAGS.sampling_rate, 
        channels = FLAGS.channels, 
        flt = FLAGS.filter, 
        exclude = FLAGS.exclude, 
        meta_regexp = FLAGS.meta_regexp, 
        max_db_size = FLAGS.max_db_size, 
        waveform = True,
    )

if __name__ == '__main__':
    app.run(main)
