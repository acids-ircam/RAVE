import sys

from absl import app

AVAILABLE_SCRIPTS = [
    'preprocess', 'train', 'train_prior', 'export', 'export_onnx', 'remote_dataset', 'generate'
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
        from scripts import train
        sys.argv[0] = train.__name__
        app.run(train.main)
    elif command == 'train_prior':
        from scripts import train_prior
        sys.argv[0] = train_prior.__name__
        app.run(train_prior.main)
    elif command == 'export':
        from scripts import export
        sys.argv[0] = export.__name__
        app.run(export.main)
    elif command == 'preprocess':
        from scripts import preprocess
        sys.argv[0] = preprocess.__name__
        app.run(preprocess.main)
    elif command == 'export_onnx':
        from scripts import export_onnx
        sys.argv[0] = export_onnx.__name__
        app.run(export_onnx.main)
    elif command == "generate":
        from scripts import generate
        sys.argv[0] = generate.__name__
        app.run(generate.main)
    elif command == 'remote_dataset':
        from scripts import remote_dataset
        sys.argv[0] = remote_dataset.__name__
        app.run(remote_dataset.main)
    else:
        raise Exception(f'Command {command} not found')
