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
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from keras.initializers import he_normal, he_uniform
from utils.chorus.satb import Satb

class ChordNet():
    @staticmethod
    def build_voice_branch(name, inputs, voice_range, final_act='softmax'):
        # x = Dense(
        #     32,
        #     kernel_initializer=he_normal(),
        #     )(inputs)
        # x = Activation('relu')(x)

        # x = Dense(
        #     32,
        #     kernel_initializer=he_uniform(),
        #     )(x)
        # x = Activation('relu')(x)

        # x = Dense(
        #     32,
        #     kernel_initializer=he_uniform(),
        #     )(x)
        # x = Activation('relu')(x)

        # x = Dense(
        #     32,
        #     kernel_initializer=he_uniform(),
        #     )(x)
        # x = Activation('relu')(x)

        # output = Dense(voice_range, activation=final_act, name=name)(x)
        output = Dense(voice_range, activation=final_act, name=name)(inputs)

        return output

    @staticmethod
    def build(input_shape=(29,), final_act='softmax'):
        inputs = Input(shape=input_shape)
        shared = Dense(
            32,
            kernel_initializer=he_normal(),
            )(inputs)        
        shared = Activation('relu')(shared)

        shared = Dense(
            32,
            kernel_initializer=he_normal(),
            )(shared)        
        shared = Activation('relu')(shared)
        
        shared = Dense(
            32,
            kernel_initializer=he_normal(),
            )(shared)   
        shared = Activation('relu')(shared)
        
        shared = Dense(
            32,
            kernel_initializer=he_normal(),
            )(shared)        
        shared = Activation('relu')(shared)

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
        
        return model