import assembler
from pieces import PiCapture, Display


def main():
    pipeline = assembler.Pipeline(
        pieces=[[PiCapture], [Display]],
        options=[[{}], [{}]]
    )
    pipeline.run()


if __name__ == '__main__':
    main()
