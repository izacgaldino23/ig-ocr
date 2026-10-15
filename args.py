import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Treinamento de rede neural')
    parser.add_argument('output',
        type=str,
        default='mnist.pth',
        nargs="?",
        help='Arquivo de saída para salvar o modelo'
    )

    args = parser.parse_args()
    
    return args