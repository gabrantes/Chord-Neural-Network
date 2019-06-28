"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/27/2019
Filename: train_val_split.py
Description: Splits the dataset (.csv) into training and validation as .txt files
"""

from utils import note_to_num
import random
import math
import argparse

def train_eval_split(input_file, percent_train=80, percent_val=20, percent_test=0):
    """
    Splits the dataset file using the given percentages. Also, reformats the data.
    Writes data to .txt files: 'train.txt', 'val.txt', 'test.txt'

    Args:   input_file -- the full filepath to the dataset
            percent_train -- percent of data allocated for training
            percent_val -- percent of data allocated for validation
            percent_test -- percent of data allocated for testing 
    """
    assert percent_train + percent_val + percent_test == 100, \
        "Percentages must add up to 100"


    chords = []
    with open(csv_file) as f:
        for line in f:
            if "//" not in line:  # ignore lines that are comments
                elements = line[:-1].split(',')  # remove newline, split elements
                chords.append([el for el in elements if el != '*'])
    f.close()

    for chord in chords:
        for i in range(len(chord)):
            if not chord[i].isdigit():  # if note
                chord[i] = note_to_num(chord[i])
            else:
                chord[i] = int(chord[i])
    # every element in chords is now an integer

    shuffle = True
    if shuffle:
        random.shuffle(chords)

    num_train = int( math.floor(len(chords) * (percent_train/100)) )
    num_val = int( math.floor(len(chords) * (percent_val/100)) )

    with open('./data/train.txt', 'w') as f:
        for chord in chords[:num_train]:
            for el in chord:
                f.write(str(el) + " ")
            f.write("\n")
    f.close()

    with open('./data/val.txt', 'w') as f:
        for chord in chords[num_train:num_val]:
            for el in chord:
                f.write(str(el) + " ")
            f.write("\n")
    f.close()

    with open('./data/test.txt', 'w') as f:
        for chord in chords[num_val:]:
            for el in chord:
                f.write(str(el) + " ")
            f.write("\n")
    f.close()

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description='Split data into train, validation, and test sets.')
    parser.add_argument("--train", type=int, help="Percentage of train set. DEFAULT: 80")
    parser.add_argument("--val", type=int, help="Percentage of validation set. DEFAULT: 20")
    parser.add_argument("--test", type=int, help="Percentage of test set. DEFAULT: 0")
    praser.add_argument("--input", help="Filepath to dataset. DEFAULT: ./data/chords.csv")

    args = parser.parse_args()

    percent_train = 80
    percent_val = 20
    percent_test = 0
    input_file = "./data/chords.csv"

    if args.train is not None:
        percent_train = args.train
    if args.val is not None:
        percent_val = args.val
    if args.test is not None:
        percent_test = args.test
    if args.input is not None:
        input_file = args.input

    train_eval_split(
        input_file,
        percent_train=percent_train,
        percent_val = percent_val,
        percent_test=percent_test
        )

