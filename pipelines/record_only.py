import assembler
from pieces import WebcamCapture, Record


def main():
    pipeline = assembler.Pipeline([
        [WebcamCapture, ],
        [Record, ],
    ])
    pipeline.run()


if __name__ == '__main__':
    main()
