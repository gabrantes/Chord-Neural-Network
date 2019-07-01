"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/30/2019
Title: data_generator.py
Description: Defines a custom generator for accessing and processing
the custom dataset. (TODO)
"""

import numpy as np

def read_data(input_file):
    """
    Handles the .txt input file and returns ndarrays, splitting up the inputs and outputs

    Returns:
        ndarray for each inputs, soprano, alto, tenor, and bass
    """
    dataset = []
    with open(input_file) as f:
        for line in f.read().splitlines():
            dataset.append([obj for obj in line.split()])
    f.close()

    dataset = np.array(dataset, dtype=np.int32)
    
    split = np.hsplit(dataset, [41])
    inputs, chord = split[0], split[1]
    split = np.hsplit(chord, 4)
    soprano, alto, tenor, bass = split[0], split[1], split[2], split[3]

    return inputs, soprano, alto, tenor, bass