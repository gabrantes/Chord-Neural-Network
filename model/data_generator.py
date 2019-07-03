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
from utils.utils import one_hot
from utils.aug import augment

def generate_prog(input_file, batch_size, aug=True):
    dataset = []
    with open(input_file) as f:
        for line in f.read().splitlines():
            dataset.append([obj for obj in line.split()])
    f.close()

    for i in range(len(dataset)):
        dataset[i] = [int(el) for el in dataset[i]]    

    count = 0
    while True:
        count = count % len(dataset)

        inputs = np.zeros((batch_size, 41))
        soprano = np.zeros((batch_size, 1))
        alto = np.zeros((batch_size, 1))
        tenor = np.zeros((batch_size, 1))
        bass = np.zeros((batch_size, 1))

        for i in range(batch_size):
            prog = dataset[count]

            if aug and np.random.randint(0, 2) == 0:  # 50% chance of augmentation
                prog = augment(prog)

            # one-hotting categories
            inputs[i, :12] = one_hot(prog[0], 12)
            inputs[i, 12] = prog[1]
            inputs[i, 13:20] = one_hot(prog[2]-1, 7)
            inputs[i, 20] = prog[3]
            inputs[i, 21:25] = one_hot(prog[4], 4)
            inputs[i, 25:29] = prog[5:9]
            inputs[i, 29:36] = one_hot(prog[9]-1, 7)
            inputs[i, 36] = prog[10]
            inputs[i, 37:] = one_hot(prog[11], 4)

            soprano[i]  = prog[12]
            alto[i]     = prog[13]
            tenor[i]    = prog[14]
            bass[i]     = prog[15]

            count += 1
        print("inputs.shape = {}".format(inputs.shape))
        yield (inputs, {'soprano': soprano, 'alto': alto, 'tenor': tenor, 'bass': bass})
    

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