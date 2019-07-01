"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/30/2019
Title: train.py
Description: Trains the neural network.
"""

from model.chordnet import ChordNet
from model.data_generator import read_data
from utils.utils import plot

from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, CSVLogger

EPOCHS = 100
BATCH_SIZE = 16

def train():
    model = ChordNet.build()

    model.summary()
    plot_model(model, to_file='chordnet.png', show_shapes=True, show_layer_names=True)

    losses = {
        "soprano": "sparse_categorical_crossentropy",
        "alto": "sparse_categorical_crossentropy",
        "tenor": "sparse_categorical_crossentropy",
        "bass": "sparse_categorical_crossentropy"
    }
    loss_weights = {
        "soprano": 1.0,
        "alto": 1.0,
        "tenor": 1.0,
        "bass": 1.0
    }

    model.compile(
        optimizer=Adam(), 
        loss=losses, 
        loss_weights=loss_weights,
        metrics=['accuracy', 'sparse_categorical_accuracy']
        )    

    checkpoint = ModelCheckpoint(
        "model.best.hdf5",
        monitor='val_loss',
        save_best_only=True
        )
    csv = CSVLogger('log.csv')
    callbacks_list = [checkpoint]

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
        verbose=1
    )

    # Plot training & validation accuracy values
    plot(H)

if __name__ == "__main__":
    train()