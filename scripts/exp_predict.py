import argparse
import numpy as np

from model.experimental_chordnet import ChordNet
from model.experimental_generator import read_data
from utils.utils import num_to_note_key
from utils.chorus.satb import Satb

def predict(input_file: str, weights: str):
    """Wrapper for model.predict with dynamic batch size, based on
    size of input file.

    Args:
        input_file: filepath to processed dataset (.txt)
    """
    # load model
    model = ChordNet.build()
    model.load_weights(weights)

    # get inputs
    inputs, next_S, next_A, next_T, next_B = read_data(input_file)
    batch_size = next_S.shape[0]

    # predict
    preds = model.predict(inputs, batch_size=batch_size)

    # process output predictions
    outs = np.zeros((batch_size, 2, 21))
    for i in range(2):
        outs[:, i, :len(preds[i][0])] = np.asarray(preds[i])   

    next_chords_num = np.zeros((batch_size, 4))
    next_chords_num[:, 0] = next_S[:, 0]
    next_chords_num[:, 3] = next_B [:, 0]
    next_chords_num[:, 1:3] = np.argmax(outs, axis=2)  

    cur_chords_num = inputs[:, 13:17]

    gt_chords_num = np.zeros((batch_size, 4))
    gt_chords_num[:, 0] = next_S[:, 0]
    gt_chords_num[:, 1] = next_A[:, 0]
    gt_chords_num[:, 2] = next_T[:, 0]
    gt_chords_num[:, 3] = next_B[:, 0]

    key_num = inputs[:, :12]
    key = np.zeros((batch_size, 2), dtype=np.int8)
    key[:, 0] = np.argmax(key_num, axis=1)  # tonic of key
    key[:, 1] = inputs[:, 12]  # major (1) or minor (0)

    next_chords =  [None] * batch_size
    cur_chords = [None] * batch_size
    gt_chords = [None] * batch_size
    
    # unscale notes, convert back to string representations
    satb = Satb()
    for i in range(batch_size):
        next_chords_num[i, :] = satb.unscale(next_chords_num[i, :].tolist())
        cur_chords_num[i, :] = satb.unscale(cur_chords_num[i, :].tolist())   
        gt_chords_num[i, :] = satb.unscale(gt_chords_num[i, :].tolist())           
        
        next_chords[i] = [num_to_note_key(int(el), key[i, 0], key[i, 1]) for el in next_chords_num[i, :].tolist()]
        cur_chords[i] = [num_to_note_key(int(el), key[i, 0], key[i, 1]) for el in cur_chords_num[i, :].tolist()]
        gt_chords[i] = [num_to_note_key(int(el), key[i, 0], key[i, 1]) for el in gt_chords_num[i, :].tolist()]

    # print predictions
    print("\n")
    for i in range(batch_size):
        if next_chords[i] != gt_chords[i]:
            out_str = "{}\t-->\t{}\tCorrect: {}\n".format(cur_chords[i], next_chords[i], gt_chords[i])
        else:
            out_str = "{}\t-->\t{}\n".format(cur_chords[i], next_chords[i])
        print(out_str)

    return next_chords

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description='Generate predictions from trained model.')
    parser.add_argument("--init",
        help="Filepath to model weights. DEFAULT: model.exp.hdf5",
        default='./model/model.exp.hdf5')
    parser.add_argument("--input", 
        help="Filepath to processed dataset. DEFAULT: ./data/test.txt",
        default="./data/test.txt")

    args = parser.parse_args()

    predict(args.input, args.init)