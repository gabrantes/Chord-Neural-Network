"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/27/2019
Title: dataGenerator.py
Description: Defines a custom generator for accessing and processing
the custom dataset.
"""

import numpy as np
import csv

def read_chords(input_file):
    """
    Handles the .csv input file, converts notes to numbers, performs
    real-time data augmentation to increase dataset size.

    Args:
        input_file: the FULL filepath to the .csv file containing the chords
    """

    # N chords progressions, 9 (5 inputs + 4 outputs)
    # chords = np.zeros(N, 9))

if __name__ == "__main__":