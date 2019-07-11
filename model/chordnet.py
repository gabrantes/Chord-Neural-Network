"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/1/2019
Filename: chordnet.py
Description:
    Defines the network architecture.
"""

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, PReLU
from keras.initializers import he_normal, he_uniform
from keras.optimizers import Adam
from utils.chorus.satb import Satb

class ChordNet():
    @staticmethod
    def build_voice_branch(name, inputs, voice_range, final_act='softmax'):
        x = Dense(
            64,
            kernel_initializer=he_normal(),
            )(inputs)

        x = PReLU()(x)

        x = Dense(
            64,
            kernel_initializer=he_uniform(),
            )(x)

        x = PReLU()(x)

        x = Dense(
            64,
            kernel_initializer=he_uniform(),
            )(x)

        x = PReLU()(x)

        x = Dense(
            16,
            kernel_initializer=he_uniform(),
            )(x)

        x = PReLU()(x)

        output = Dense(voice_range, activation=final_act, name=name)(x)

        return output

    @staticmethod
    def build(input_shape=(29,), final_act='softmax'):
        inputs = Input(shape=input_shape)
        shared = Dense(
            128,
            kernel_initializer=he_normal(),
            )(inputs)
        
        shared = PReLU()(shared)

        # shared = Dense(
        #     128,
        #     kernel_initializer=he_normal(),
        #     activation='relu'
        #     )(shared)

        shared = Dense(
            128, 
            kernel_initializer=he_normal()
            )(shared)        
        shared = BatchNormalization(scale=False)(shared)
        shared = PReLU()(shared)

        satb = Satb()
        soprano = ChordNet.build_voice_branch(
            'soprano',
            shared,
            satb.voices[0].range,
            final_act=final_act
            )
        alto = ChordNet.build_voice_branch(
            'alto',
            shared,
            satb.voices[1].range,
            final_act=final_act
            )
        tenor = ChordNet.build_voice_branch(
            'tenor',
            shared,
            satb.voices[2].range,
            final_act=final_act
            )
        bass = ChordNet.build_voice_branch(
            'bass',
            shared,
            satb.voices[3].range,
            final_act=final_act
            )

        model = Model(
            inputs=inputs,
            outputs=[soprano, alto, tenor, bass],
            name="ChordNet"
            )
        
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
            metrics=['sparse_categorical_accuracy']
            )
        
        return model