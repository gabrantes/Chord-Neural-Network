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
from model.data_generator import read_data
from model.train_val_tensorboard import TrainValTensorBoard

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
    callbacks_list = [checkpoint, tensorboard]

    train_X, train_S, train_A, train_T, train_B = read_data("./data/train.txt")
    val_X, val_S, val_A, val_T, val_B = read_data("./data/val.txt")

    H = model.fit(
        train_X,
        {"soprano": train_S, "alto": train_A, "tenor": train_T, "bass": train_B},
        validation_data=(
            val_X,
            {"soprano": val_S, "alto": val_A, "tenor": val_T, "bass": val_B}
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=2
    )

if __name__ == "__main__":
    train()