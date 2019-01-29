"""
train.py

1. load data
2. define or load model
3. train
"""
import os, sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.logger import logger
from utils.parser import parse_args
from data.dataset import Dataset
from models.fcn import fcn_8s

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # define logger
    base_path = os.path.abspath('..')
    logger = logger(name='train', path=os.path.join(base_path, 'log/train.log'))

    # parse argument
    logger.info("Load Arguments")
    args = parse_args()
    logger.info(args)

    model_name = args.model
    vgg_path = args.vgg
    image_size = args.image_size
    lr_init = args.lr_init
    lr_decay = args.lr_decay

    # load data
    logger.info("Load Data")

    data_path = os.path.join(base_path, 'data/dataset1/train')
    image_path = os.path.join(data_path, 'images')
    mask_path = os.path.join(data_path, 'masks')

    data = Dataset(image_path=image_path, mask_path=mask_path, image_size=image_size[:2], logger=logger)
    logger.info("image" + str(data.images.shape))
    logger.info("mask" + str(data.masks.shape))

    # model
    logger.info("Set Parameter for train")
    num_classes = 3
    epochs = 100

    keras_path = os.path.join(base_path, 'keras')
    weight_path = os.path.join(keras_path, 'weight')
    graph_path = os.path.join(keras_path, 'graph')

    checkpoint = ModelCheckpoint(filepath=os.path.join(weight_path, model_name + '_model_weight.h5'),
                                 monitor='val_dice_coef',
                                 save_best_only=True,
                                 save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_dice_coef', patience=10)

    tensorboard = TensorBoard(log_dir=weight_path,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    if model_name == "fcn":
        model = fcn_8s(input_shape=image_size,
                       num_classes=num_classes,
                       lr_init=lr_init,
                       lr_decay=lr_decay,
                       vgg_weight_path=vgg_path)

    logger.info("Starts training")
    history = model.fit(x=data.images,
                        y=data.masks,
                        epochs=epochs,
                        validation_split=0.2,
                        verbose=2)

    logger.info("Print out result")
    plt.title("loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="val")
    plt.legend(loc="best")
    plt.savefig(model_name + '_loss.png')

    plt.gcf().clear()
    plt.title("dice_coef")
    plt.plot(history.history["dice_coef"], color="r", label="train")
    plt.plot(history.history["val_dice_coef"], color="b", label="val")
    plt.legend(loc="best")

    result_path = os.path.join(keras_path, 'result')
    plt.savefig(os.path.join(result_path, model_name + '_dice_coef.png'))















