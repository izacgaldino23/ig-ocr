import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Treinamento de rede neural")
    parser.add_argument("-i", "--input", type=int, default=0, help="Modelo número")

    args = parser.parse_args()

    return args
