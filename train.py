"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/1/2019
Title: train.py
Description: Trains the neural network.
"""

import time
import os

from model.chordnet import ChordNet
from model.train_val_tensorboard import TrainValTensorBoard
from model.dead_relu_detector import DeadReluDetector
from model.data_generator import generate_prog, read_data

from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

EPOCHS = 100
BATCH_SIZE = 16

def train():
    model = ChordNet.build()
    model.summary()

    cur_time = time.localtime()
    log_dir = "./logs/{}.{}.{}{}".format(
            cur_time[1],
            cur_time[2],
            cur_time[3],
            cur_time[4]
        )
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    plot_model(model, to_file=log_dir+'/ChordNet.png', show_shapes=True, show_layer_names=True)  

    checkpoint = ModelCheckpoint(
        "model.best.hdf5",
        monitor='val_loss',
        save_best_only=True
        )
    
    tensorboard = TrainValTensorBoard(
        log_dir=log_dir,
        write_graph=True,
        histogram_freq=10,
        write_grads=True
        )    

    train_X, train_S, train_A, train_T, train_B = read_data("./data/train.txt")
    val_X, val_S, val_A, val_T, val_B = read_data("./data/val.txt")

    print("val_X.shape = {}".format(val_X.shape))
    assert val_X.shape[1] == 41
    assert val_S.shape[1] == 1
    assert val_A.shape[1] == 1
    assert val_T.shape[1] == 1
    assert val_B.shape[1] == 1

    dead_relu_detector = DeadReluDetector(x_train=train_X)

    callbacks_list = [checkpoint, tensorboard, dead_relu_detector]

    train_gen = generate_prog("./data/train.txt", BATCH_SIZE, aug=True)

    H = model.fit_generator(
        train_gen,
        steps_per_epoch = 1971 // BATCH_SIZE,
        epochs = EPOCHS,
        verbose = 1,
        callbacks = callbacks_list,
        validation_data = (
            val_X,
            {"soprano": val_S, "alto": val_A, "tenor": val_T, "bass": val_B}
        )
    )

if __name__ == "__main__":
    train()