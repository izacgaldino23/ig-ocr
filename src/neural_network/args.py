import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Neural network training")
    parser.add_argument("-i", "--input", type=float, default=0, help="Input model number")
    parser.add_argument("-o", "--output", type=float, default=0, help="Output model number")

    args = parser.parse_args()

    return args
