import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Treinamento de rede neural')
    parser.add_argument('-o', '--output',
        type=str,
        default='mnist.pth',
        help='Arquivo de saída para salvar o modelo'
    )
    parser.add_argument('-i', '--input',
        type=str,
        default='mnist.pth',
        help='Arquivo de entrada para carregar o modelo'
    )

    args = parser.parse_args()
    
    return args