"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/2/2019
Filename: predict.py
Description: 
    Generate predictions form trained model.
"""

import argparse
import numpy as np

from model.chordnet import ChordNet
from utils.utils import num_to_note
from utils.satb import Satb

def predict(input_file: str, weights: str):
    """Wrapper for model.predict with dynamic batch size, based on
    size of inputs.

    Args:
        input_file: filepath to processed dataset (.txt)
    """
    model = ChordNet.build()
    model.load_weights(weights)

    with open(input_file) as f:
        lines = f.read().splitlines()
    f.close()

    batch_size = len(lines)
    inputs = np.zeros((batch_size, 41), dtype=np.int16)
    for i in range(batch_size):
        inputs[i, :] = [int(el) for el in lines[i].split()[:41]]

    satb = Satb()

    preds = model.predict(inputs, batch_size=batch_size)

    outs = np.zeros((batch_size, 4, 21))
    for i in range(4):
        outs[:, i, :len(preds[i][0])] = np.asarray(preds[i])        
    next_chords_num = np.argmax(outs, axis=2)    
    cur_chords_num = inputs[:, 25:29]

    next_chords =  [None] * batch_size
    cur_chords = [None] * batch_size
    
    for i in range(batch_size):
        next_chords_num[i, :] = satb.unscale(next_chords_num[i, :].tolist())
        cur_chords_num[i, :] = satb.unscale(cur_chords_num[i, :].tolist())              
        
        next_chords[i] = [num_to_note(int(el)) for el in next_chords_num[i, :].tolist()]
        cur_chords[i] = [num_to_note(int(el)) for el in cur_chords_num[i, :].tolist()]


    for i in range(batch_size):
        out_str = "{}\t-->\t{}".format(cur_chords[i], next_chords[i])
        print(out_str)

    return preds

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description='Generate predictions from trained model.')
    parser.add_argument("--init",
        help="Filepath to model weights. DEFAULT model.best.hdf5",
        default='model.best.hdf5')
    parser.add_argument("--input", 
        help="Filepath to processed dataset. DEFAULT: ./data/test.txt",
        default="./data/test.txt")

    args = parser.parse_args()

    predict(args.input, args.init)

