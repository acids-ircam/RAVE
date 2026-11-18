import sys

from absl import app

AVAILABLE_SCRIPTS = [
    'preprocess', 'train', 'finetune', 'train_prior', 'train_after', 'export', 'export_onnx', 'remote_dataset', 'generate', 'adapt', 'uncompile', 'dimred'
]


def help():
    print(f"""usage: rave [ {' | '.join(AVAILABLE_SCRIPTS)} ]

positional arguments:
  command     Command to launch with rave.
""")
    exit()


def main():
    if len(sys.argv) == 1:
        help()
    elif sys.argv[1] not in AVAILABLE_SCRIPTS:
        help()

    command = sys.argv[1]

    if command == 'train':
        from rave.scripts import train
        sys.argv[0] = train.__name__
        app.run(train.main)
    elif command == 'finetune':
        from rave.scripts import finetune
        sys.argv[0] = finetune.__name__
        app.run(finetune.main)
    elif command == 'adapt':
        from rave.scripts import adapt
        sys.argv[0] = adapt.__name__
        app.run(adapt.main)
    elif command == 'train_prior':
        from rave.scripts import train_prior
        sys.argv[0] = train_prior.__name__
        app.run(train_prior.main)
    elif command == 'train_after':
        from rave.scripts import train_after
        train_after.parse_flags()  # flags are not defined at import
        sys.argv[0] = train_after.__name__
        app.run(train_after.main)
    elif command == 'export':
        from rave.scripts import export
        sys.argv[0] = export.__name__
        app.run(export.main)
    elif command == 'preprocess':
        from rave.scripts import preprocess
        sys.argv[0] = preprocess.__name__
        app.run(preprocess.main)
    elif command == 'export_onnx':
        from rave.scripts import export_onnx
        sys.argv[0] = export_onnx.__name__
        app.run(export_onnx.main)
    elif command == "generate":
        from rave.scripts import generate
        sys.argv[0] = generate.__name__
        app.run(generate.main)
    elif command == 'remote_dataset':
        from rave.scripts import remote_dataset
        sys.argv[0] = remote_dataset.__name__
        app.run(remote_dataset.main)
    elif command == 'uncompile':
        from rave.scripts import uncompile 
        sys.argv[0] = uncompile.__name__
        app.run(uncompile.main)
    elif command == 'dimred':
        from rave.scripts import dimred
        sys.argv[0] = dimred.__name__
        app.run(dimred.main)
    else:
        raise Exception(f'Command {command} not found')
