from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Activation
from keras.initializers import he_normal, he_uniform
from keras.optimizers import Adam
from utils.chorus.satb import Satb

class ChordNet():
    @staticmethod
    def build_voice_branch(name, inputs, voice_range, final_act='softmax'):
        x = Dense(
            64,
            kernel_initializer=he_normal(),
            activation='relu'
            )(inputs)

        x = Dense(
            64,
            kernel_initializer=he_uniform(),
            activation='relu'
            )(x)

        x = Dense(
            64,
            kernel_initializer=he_uniform(),
            activation='relu'
            )(x)

        x = Dense(
            16,
            kernel_initializer=he_uniform(),
            activation='relu'
        )(x)

        output = Dense(voice_range, activation=final_act, name=name)(x)

        return output

    @staticmethod
    def build(input_shape=(19,), final_act='softmax'):
        inputs = Input(shape=input_shape)

        shared = Dense(
            128,
            kernel_initializer=he_normal(),
            activation='relu'
            )(inputs)

        shared = Dense(
            128, 
            kernel_initializer=he_uniform(),
            activation='relu'
            )(shared)

        shared = Dense(
            128, 
            kernel_initializer=he_uniform(),
            )(shared) 

        shared = BatchNormalization(scale=False)(shared)
        shared = Activation('relu')(shared)

        satb = Satb()
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

        model = Model(
            inputs=inputs,
            outputs=[alto, tenor],
            name="Experimental_ChordNet"
            )
        
        losses = {
            "alto": "sparse_categorical_crossentropy",
            "tenor": "sparse_categorical_crossentropy"
        }
        loss_weights = {
            "alto": 1.0,
            "tenor": 1.0
        }

        model.compile(
            optimizer=Adam(), 
            loss=losses, 
            loss_weights=loss_weights,
            metrics=['sparse_categorical_accuracy']
            )
        
        return model