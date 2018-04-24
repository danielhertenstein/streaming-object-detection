import assembler
from pieces import WebcamCapture, Record


def main():
    pipeline = assembler.Pipeline(
        pieces=[[WebcamCapture], [Record]],
        args=[[{}], [{}]]
    )
    pipeline.run()


if __name__ == '__main__':
    main()
