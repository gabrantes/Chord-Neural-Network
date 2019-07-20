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
import pandas as pd

from model.chordnet import ChordNet
from utils.utils import num_to_note_key, num_to_note
from utils.chorus.satb import Satb
from keras.models import load_model

def predict(input_file: str, model: str):
    """Wrapper for model.predict with dynamic batch size, based on
    size of input file.

    Args:
        input_file: filepath to processed dataset (.txt)
    """
    # load model
    print("Loading model from {}".format(model))
    model = load_model(model)

    # get inputs
    with open(input_file) as f:
        lines = f.read().splitlines()
    f.close()

    batch_size = len(lines)
    inputs = np.zeros((batch_size, 45), dtype=np.int16)
    for i in range(batch_size):
        inputs[i, :] = [int(el) for el in lines[i].split()]  

    feat_inputs = np.zeros((batch_size, 29))  
    feat_inputs[:, 13] = inputs[:, 13]
    feat_inputs[:, 13:17] = inputs[:, 25:29]
    feat_inputs[:, 17:29] = inputs[:, 29:41]

    # predict
    pred_s, pred_a, pred_t, pred_b = model.predict(feat_inputs, batch_size=batch_size)

    # process output predictions
    outs = np.zeros((batch_size, 4, 21))

    outs[:, 0, :len(pred_s[0])] = np.asarray(pred_s)
    outs[:, 1, :len(pred_a[0])] = np.asarray(pred_a)
    outs[:, 2, :len(pred_t[0])] = np.asarray(pred_t)
    outs[:, 3, :len(pred_b[0])] = np.asarray(pred_b)
    # outs = (batch_size, voice, note)
     
    next_chords_num = np.argmax(outs, axis=2)  

    cur_chords_num = inputs[:, 25:29]
    gt_chords_num = inputs[:, 41:]

    key_num = inputs[:, :12]
    key = np.zeros((batch_size, 2), dtype=np.int8)
    key[:, 0] = np.argmax(key_num, axis=1)  # tonic of key
    key[:, 1] = inputs[:, 12]  # major (1) or minor (0)

    next_chords =  [None] * batch_size
    cur_chords = [None] * batch_size
    gt_chords = [None] * batch_size
    key_note = [None] * batch_size
    notes_correct = [None] * batch_size
    
    # unscale notes, convert back to string representations
    satb = Satb()
    for i in range(batch_size):
        try:
            next_chords_num[i, :] = satb.unscale(next_chords_num[i, :].tolist())
            next_chords[i] = [num_to_note_key(int(el), key[i, 0], key[i, 1]) for el in next_chords_num[i, :].tolist()]
        except:
            next_chords[i] = "INVALID"
        cur_chords_num[i, :] = satb.unscale(cur_chords_num[i, :].tolist())  
        gt_chords_num[i, :] = satb.unscale(gt_chords_num[i, :].tolist())            
        
        
        cur_chords[i] = [num_to_note_key(int(el), key[i, 0], key[i, 1]) for el in cur_chords_num[i, :].tolist()]
        gt_chords[i] = [num_to_note_key(int(el), key[i, 0], key[i, 1]) for el in gt_chords_num[i, :].tolist()]

        if next_chords[i] == "INVALID":
            notes_correct[i] = 0
        else:
            count = 0
            for j in range(4):
                if next_chords[i][j] == gt_chords[i][j]:
                    count += 1
            notes_correct[i] = count

        key_note[i] = num_to_note(key[i, 0])
    
    # print predictions
    df = pd.DataFrame(columns=[
        'key', 'maj/min', 'cur_chord', 
        'next_deg', 'next_sev', 'next_inv',
        'pred_next', 'gt_next', 'notes_correct'
        ])

    df['key'] = key_note
    df['maj/min'] = key[:, 1]
    df['cur_chord'] = cur_chords
    df['next_deg'] = np.argmax(feat_inputs[:, 17:24], axis=1) + 1
    df['next_sev'] = feat_inputs[:, 24].astype('uint8')
    df['next_inv'] = np.argmax(feat_inputs[:, 25:29], axis=1)
    df['pred_next'] = next_chords
    df['gt_next'] = gt_chords
    df['notes_correct'] = notes_correct

    print(df.to_string())

    print("\nNotes correct:\t{}".format(sum(notes_correct)))

    df.to_excel('output.xlsx')

    return next_chords

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description='Generate predictions from trained model.')
    parser.add_argument("--init",
        help="Filepath to model. DEFAULT: model.feat.hdf5",
        default='./model/model.feat.hdf5')
    parser.add_argument("--input", 
        help="Filepath to processed dataset. DEFAULT: ./data/test.txt",
        default="./data/test.txt")

    args = parser.parse_args()

    predict(args.input, args.init)

