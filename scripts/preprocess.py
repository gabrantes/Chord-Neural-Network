"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/2/2019
Filename: preprocess.py
Description: 
    Preprocess dataset, augments, and splits into training and validation as .txt files
"""

import random
import math
import argparse
import numpy as np
from utils.utils import note_to_num, num_to_note, one_hot
from utils.aug import augment
from utils.chorus.satb import Satb

VERBOSE = False
OUTPUT_DIR = './data'

def train_eval_split(input_file: str, percent_train=80, percent_val=20, percent_test=0):
    """
    Reformat and augment the dataset, before splitting using the given percentages.
    Writes data to .txt files: 'train.txt', 'val.txt', 'test.txt'

    Args:   
        input_file: the full filepath to the dataset
        percent_train: percent of data allocated for training
        percent_val: percent of data allocated for validation
        percent_test: percent of data allocated for testing 
    """
    assert percent_train + percent_val + percent_test == 100, \
        "Percentages must add up to 100"

    # getting chord progressions from input file
    progressions = []
    with open(input_file) as f:
        for line in f:
            if "//" not in line:  # ignore lines that are comments
                elements = line[:-1].split(',')  # remove newline, split elements
                progressions.append([el for el in elements if el != '*'])
    f.close()

    # convert all notes to ints
    for prog in progressions:
        for i in range(len(prog)):
            if not prog[i].isdigit():  # if note
                prog[i] = note_to_num(prog[i])
            else:
                prog[i] = int(prog[i])

   
    satb = Satb()
    for i in range(len(progressions)):
        # scale all chords
        progressions[i][5:9] = satb.scale(progressions[i][5:9])
        progressions[i][12:16] = satb.scale(progressions[i][12:16])

        # one-hot the tonic / key-signature
        progressions[i][0] = tuple(one_hot(progressions[i][0], 12))

        # one-hot the scale-degrees
        progressions[i][2] = tuple(one_hot(progressions[i][2]-1, 7))
        progressions[i][9] = tuple(one_hot(progressions[i][9]-1, 7))

        # one-hot the inversions
        progressions[i][4] = tuple(one_hot(progressions[i][4], 4))
        progressions[i][11] = tuple(one_hot(progressions[i][11], 4))
        
    # remove duplicates and keep track of unique progressions using a set
    progression_set = set([tuple(prog) for prog in progressions])
    progressions = [list(prog) for prog in progression_set]
    
    beg_num_chords = len(progressions)
    beg_inv_chords = 0
    beg_sev_chords = 0

    duplicate_chords = 0

    end_num_chords = beg_num_chords
    end_inv_chords = 0
    end_sev_chords = 0

    for i in range(len(progressions)):
        if VERBOSE:
            print(i)        

        aug_count = 10  # number of augmentations to create
        sev_chord = False
        inv_chord = False
        if progressions[i][3] == 1 or progressions[i][10] == 1:  # if seventh chord           
            # aug_count = 10
            beg_sev_chords += 1
            end_sev_chords += 1
            sev_chord = True
        if progressions[i][4][0] == 0 or progressions[i][11][0] == 0:  # if chord not in root-position
            # aug_count = 20
            beg_inv_chords += 1
            end_inv_chords += 1
            inv_chord = True

        # data augmentation
        for _ in range(aug_count):
            new_prog = augment(progressions[i])

            if tuple(new_prog) not in progression_set:
                progressions.append(new_prog)
                progression_set.add(tuple(new_prog))
                end_num_chords += 1
                if inv_chord:
                    end_inv_chords += 1
                if sev_chord:
                    end_sev_chords += 1
            else:
                duplicate_chords += 1
                continue

            if VERBOSE:
                print("Before:")
                print("\t{}".format(num_to_note(progressions[i][0].index(1))))
                print("\t{}".format([num_to_note(el) for el in satb.unscale(progressions[i][5:9])]))
                print("\t{}".format([num_to_note(el) for el in satb.unscale(progressions[i][12:16])]))
                print("After:")
                print("\t{}".format(num_to_note(new_prog[0].index(1))))
                print("\t{}".format([num_to_note(el) for el in satb.unscale(new_prog[5:9])]))
                print("\t{}".format([num_to_note(el) for el in satb.unscale(new_prog[12:16])]))
                print("\n")

    shuffle = True
    if shuffle:
        random.shuffle(progressions)

    num_train = int(len(progressions) * (percent_train/100))
    num_val = int(len(progressions) * (percent_val/100))

    # write to files
    with open(OUTPUT_DIR + '/train.txt', 'w') as f:
        for prog in progressions[:num_train]:
            for el in prog:
                if isinstance(el, tuple):
                    for e in el:
                        f.write(str(e) + " ")
                else:
                    f.write(str(el) + " ")
            f.write("\n")
    f.close()

    num_val += num_train  # for the list slicing
    with open(OUTPUT_DIR + '/val.txt', 'w') as f:        
        for prog in progressions[num_train:num_val]:
            for el in prog:
                if isinstance(el, tuple):
                    for e in el:
                        f.write(str(e) + " ")
                else:
                    f.write(str(el) + " ")
            f.write("\n")
    f.close()

    with open(OUTPUT_DIR + '/test.txt', 'w') as f:
        for prog in progressions[num_val:]:
            for el in prog:
                if isinstance(el, tuple):
                    for e in el:
                        f.write(str(e) + " ")
                else:
                    f.write(str(el) + " ")
            f.write("\n")
    f.close()
    num_val -= num_train  # back to actual value

    print("\n************")

    print("\nSTART:")
    print("Total number of progressions:\t\t{}".format(beg_num_chords))
    print("Number of prog. with inverted chords:\t{}\t{}".format(beg_inv_chords, beg_inv_chords/beg_num_chords))
    print("Number of prog. with seventh chords:\t{}\t{}".format(beg_sev_chords, beg_sev_chords/beg_num_chords))

    print("\nDuplicate progressions:\t\t{}".format(duplicate_chords))
    print("(created during augmentation and removed)")
    
    print("\nEND:")
    print("Total number of progressions:\t\t{}".format(end_num_chords))
    print("Number of prog. with inverted chords:\t{}\t{}".format(end_inv_chords, end_inv_chords/end_num_chords))
    print("Number of prog. with seventh chords:\t{}\t{}".format(end_sev_chords, end_sev_chords/end_num_chords))

    print("\nNumber in train:\t{}".format(num_train))
    print("Number in val:\t\t{}".format(num_val))
    print("Number in test:\t\t{}\n".format(len(progressions)-num_train-num_val))  

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description='Split data into train, validation, and test sets.')
    parser.add_argument("--train", type=float, 
        help="Percentage of train set. DEFAULT: 80",
        default=80)
    parser.add_argument("--val", type=float, 
        help="Percentage of validation set. DEFAULT: 19",
        default=19)
    parser.add_argument("--test", type=float, 
        help="Percentage of test set. DEFAULT: 1",
        default=1)
    parser.add_argument("--input", 
        help="Filepath to dataset. DEFAULT: ./data/chords.csv",
        default="./data/chords.csv")
    parser.add_argument("--output", 
        help="The output directory to save the resulting .txt files. DEFAULT: ./data/",
        default="./data/")
    parser.add_argument("-v", "--verbose",
        help="Prints out chords for debugging. DEFAULT: False",
        action='store_true')

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True
    OUTPUT_DIR = args.output

    train_eval_split(
        args.input,
        percent_train=args.train,
        percent_val=args.val,
        percent_test=args.test
        )