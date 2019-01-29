"""
train.py

1. logger and parser
2. load data
3. define or load a model
4. train
5. save result
"""
import os, sys

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from matplotlib import pyplot as plt

from utils.logger import logger
from utils.parser import parse_args

from data.dataset import Dataset
from models.fcn import fcn_8s


if __name__ == "__main__":

    # define logger
    base_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_path, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logger(name='train', path=os.path.join(log_path, 'train.log'))

    # parse argument
    logger.info("Load Arguments")
    args = parse_args()
    logger.info(args)

    model_name = args.model
    vgg_path = args.vgg
    image_size = args.image_size

    batch_size = args.batch_size
    epochs = args.epochs
    lr_init = args.lr_init
    lr_decay = args.lr_decay

    # load data
    logger.info("Load Data")

    data_path = os.path.join(base_path, 'data/dataset1/train')
    image_path = os.path.join(data_path, 'images')
    mask_path = os.path.join(data_path, 'masks')

    data = Dataset(image_path=image_path, mask_path=mask_path, image_size=image_size, logger=logger)
    logger.info("image" + str(data.images.shape))
    logger.info("mask" + str(data.masks.shape))

    # model
    logger.info("Set Parameter for train")
    num_classes = 3

    weight_path = os.path.join(base_path, 'weight')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    graph_path = os.path.join(base_path, 'graph')
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    checkpoint = ModelCheckpoint(filepath=os.path.join(weight_path, model_name + '_model_weight.h5'),
                                 monitor='val_dice_coef',
                                 save_best_only=True,
                                 save_weights_only=True)
    early_stopping = EarlyStopping(monitor='val_dice_coef', patience=5)

    tensorboard = TensorBoard(log_dir=graph_path,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    if model_name == "fcn":
        model = fcn_8s(input_shape=image_size,
                       num_classes=num_classes,
                       lr_init=lr_init,
                       lr_decay=lr_decay,
                       vgg_weight_path=vgg_path)
    elif model_name == "fcn_small":
        model = fcn_8s(input_shape=image_size,
                       num_classes=num_classes,
                       lr_init=lr_init,
                       lr_decay=lr_decay)
    else:
        pass

    logger.info("Starts training")
    history = model.fit(x=data.images,
                        y=data.masks,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[checkpoint, early_stopping, tensorboard],
                        verbose=1)

    logger.info("save result")

    result_path = os.path.join(base_path, 'result')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    plt.title("loss")
    plt.plot(history.history["loss"], color="r", label="train")
    plt.plot(history.history["val_loss"], color="b", label="val")
    plt.legend(loc="best")
    plt.savefig(os.path.join(result_path, model_name + '_loss.png'))

    plt.gcf().clear()
    plt.title("dice_coef")
    plt.plot(history.history["dice_coef"], color="r", label="train")
    plt.plot(history.history["val_dice_coef"], color="b", label="val")
    plt.legend(loc="best")
    plt.savefig(os.path.join(result_path, model_name + '_dice_coef.png'))
