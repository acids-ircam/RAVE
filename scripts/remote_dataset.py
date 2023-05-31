import base64
import logging
import os

import flask
import numpy as np
from absl import flags
from udls import AudioExample

from rave.dataset import get_dataset

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "db_path",
    default=None,
    required=True,
    help="path to database.",
)
flags.DEFINE_integer(
    "sr",
    default=44100,
    help="sampling rate.",
)
flags.DEFINE_integer(
    "n_signal",
    default=2**16,
    help="sample size.",
)
flags.DEFINE_integer(
    "port",
    default=5000,
    help="port to serve the dataset.",
)


def main(argv):
    app = flask.Flask(__name__)
    dataset = get_dataset(db_path=FLAGS.db_path,
                          sr=FLAGS.sr,
                          n_signal=FLAGS.n_signal)

    @app.route("/")
    def main():
        return ("<h1>RAVE remote dataset</h1>\n"
                f"<p>Serving: {os.path.abspath(FLAGS.db_path)}</p>\n"
                f"<p>Length: {len(dataset)}</p>")

    @app.route("/len")
    def length():
        return flask.jsonify(len(dataset))

    @app.route("/get/<index>")
    def get(index):
        index = int(index)
        ae = AudioExample()
        ae.put("audio", dataset[index], np.float32)
        ae = base64.b64encode(bytes(ae))
        return ae

    app.run(host="0.0.0.0", port=FLAGS.port)
