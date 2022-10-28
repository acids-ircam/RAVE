import sys
from absl import app


def help():
    print("""usage: rave [ preprocess | train | export ]

positional arguments:
  command     Command to launch with rave. Must be either preprocess, train or export.
""")
    exit()


def main():
    if len(sys.argv) == 1:
        help()
    elif sys.argv[1] not in ['preprocess', 'train', 'export']:
        help()

    command = sys.argv[1]

    if command == 'train':
        from scripts import train
        sys.argv[0] = train.__name__
        app.run(train.main)
    elif command == 'export':
        from scripts import export
        sys.argv[0] = export.__name__
        app.run(export.main)
    elif command == 'preprocess':
        from scripts import preprocess
        sys.argv[0] = preprocess.__name__
        app.run(preprocess.main)
    else:
        raise Exception(f'Command {command} not found')
