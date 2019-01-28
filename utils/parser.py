import argparse


def parse_args():
    """
    get argument for segmnetation

    :return: args: arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for segmentation')
    parser.add_argument('--model', type=str, help="name of the model")
    args = parser.parse_args()
    return args

