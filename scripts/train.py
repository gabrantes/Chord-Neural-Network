"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/2/2019
Filename: train.py
Description: 
    Trains the neural network.
"""

import time
import os

from model.chordnet import ChordNet
from model.data_generator import generate_prog, read_data
from utils.train_val_tensorboard import TrainValTensorBoard

from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

EPOCHS = 100
BATCH_SIZE = 32

def train():
    model = ChordNet.build()
    model.summary()

    cur_time = time.localtime()
    log_dir = "./logs/feature_selection/{}.{}.{}{}".format(
            cur_time[1],
            cur_time[2],
            cur_time[3],
            cur_time[4]
        )
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    plot_model(model, to_file=log_dir+'/ChordNet.png', show_shapes=True, show_layer_names=True)  

    checkpoint = ModelCheckpoint(
        "model/model.feat.hdf5",
        monitor='val_loss',
        save_best_only=True
        )
    
    tensorboard = TrainValTensorBoard(
        log_dir=log_dir,
        write_graph=True,
        )    

    num_train_data = 0
    with open("./data/train.txt") as f:
        num_train_data = len(f.readlines())
    f.close()

    val_X, val_S, val_A, val_T, val_B = read_data("./data/val.txt")

    callbacks_list = [checkpoint, tensorboard]

    train_gen = generate_prog("./data/train.txt", BATCH_SIZE, aug=True)

    H = model.fit_generator(
        train_gen,
        steps_per_epoch = num_train_data // BATCH_SIZE,
        epochs = EPOCHS,
        verbose = 2,
        callbacks = callbacks_list,
        validation_data = (
            val_X,
            {"soprano": val_S, "alto": val_A, "tenor": val_T, "bass": val_B}
        )
    )

if __name__ == "__main__":
    train()