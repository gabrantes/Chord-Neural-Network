"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/30/2019
Title: chordnet.py
Description: Defines the network architecture
"""

from keras.models import Model
from keras.layers import Input, Dense, Activation
from utils.satb import Satb

class ChordNet():
    @staticmethod
    def build_voice_branch(name, inputs, voice_range, final_act='softmax'):
        dense_1 = Dense(64, activation='relu')(inputs)
        dense_2 = Dense(64, activation='relu')(dense_1)
        dense_3 = Dense(64, activation='relu')(dense_2)
        dense_4 = Dense(16, activation='relu')(dense_3)
        output = Dense(voice_range, activation=final_act, name=name)(dense_4)

        return output

    @staticmethod
    def build(input_shape=(41,), final_act='softmax'):
        inputs = Input(shape=input_shape)
        shared_1 = Dense(128, activation='relu')(inputs)
        shared_2 = Dense(128, activation='relu')(shared_1)

        satb = Satb()
        soprano = ChordNet.build_voice_branch(
            'soprano',
            shared_2,
            satb.voices[0].range,
            final_act=final_act)
        alto = ChordNet.build_voice_branch(
            'alto',
            shared_2,
            satb.voices[1].range,
            final_act=final_act)
        tenor = ChordNet.build_voice_branch(
            'tenor',
            shared_2,
            satb.voices[2].range,
            final_act=final_act)
        bass = ChordNet.build_voice_branch(
            'bass',
            shared_2,
            satb.voices[3].range,
            final_act=final_act)

        model = Model(
            inputs=inputs,
            outputs=[soprano, alto, tenor, bass],
            name="ChordNet"
            )
        
        return model