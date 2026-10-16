import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Treinamento de rede neural")
    parser.add_argument("-i", "--input", type=float, default=0, help="Número do modelo para entrada")
    parser.add_argument("-o", "--output", type=float, default=0, help="Número do modelo para saida")

    args = parser.parse_args()

    return args
