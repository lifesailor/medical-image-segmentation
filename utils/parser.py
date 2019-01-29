import argparse


def parse_args():
    """
    get argument for segmnetation

    :return: args: arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for segmentation')
    parser.add_argument("-M", "--model", required=True, choices=['fcn', 'fcn_small', 'unet', 'pspnet'],
                        help="Model to train. 'fcn', 'unet', 'pspnet' is available.")
    parser.add_argument("--vgg", required=False, default=None, help="Pretrained vgg16 weight path.")
    parser.add_argument("-IS", "--image_size", default=(256, 256, 3), help="input image size.")
    parser.add_argument("-BS", "--batch_size", required=False, default=32, help="Initial learning rate.")
    parser.add_argument("-E", "--epochs", required=False, default=30, help="How much to decay the learning rate.")
    parser.add_argument("-LI", "--lr_init", required=False, default=1e-3, help="Initial learning rate.")
    parser.add_argument("-LD", "--lr_decay", required=False, default=5e-4, help="How much to decay the learning rate.")
    args = parser.parse_args()
    return args
