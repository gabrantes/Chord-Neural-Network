import numpy as np
from utils.utils import one_hot
from utils.aug import augment

def generate_prog(input_file, batch_size, aug=True):
    """Custom generator for use with Keras fit_generator.
    Performs real-time data augmentation by transposing chord progressions
    to random keys.

    Args:
        input_file: filepath to the PRE-PROCESSED input files (.txt)
        batch_size: batch size
        aug:        if True, perform real-time data augmentation.

    Yields:
        inputs: ndarray of size (batch_size, 41)
        soprano, alto, tenor, bass: ndarrays of size (batch_size, 1)
    """
    dataset = []
    with open(input_file) as f:
        for line in f.read().splitlines():
            dataset_row = []
            line = [int(obj) for obj in line.split()]
            dataset_row.append(tuple([el for el in line[:12]]))  # key
            dataset_row.append(line[12])

            dataset_row.append(tuple([el for el in line[13:20]]))  # cur_degree
            dataset_row.append(line[20])  # cur_sev
            dataset_row.append(tuple([el for el in line[21:25]]))  # cur_inv
            dataset_row.extend(line[25:29])  # cur_chord

            dataset_row.append(tuple([el for el in line[29:36]]))  # next_degree
            dataset_row.append(line[36])  # next_sev
            dataset_row.append(tuple([el for el in line[37:41]]))  # next_inv
            dataset_row.extend(line[41:])  # next_chord

            dataset.append(dataset_row)
    f.close()

    count = 0
    while True:
        inputs = np.zeros((batch_size, 19))
        alto = np.zeros((batch_size, 1))
        tenor = np.zeros((batch_size, 1))

        for i in range(batch_size):
            count = count % len(dataset)
            prog = dataset[count]
            assert len(prog) == 16

            if aug and np.random.randint(0, 100) < 75:  # 75% chance of augmentation
                prog = augment(prog)

            # one-hotting categories
            inputs[i, :12] = list(prog[0])  # key
            inputs[i, 12] = prog[1]
            # inputs[i, 13:20] = list(prog[2])
            # inputs[i, 20] = prog[3]
            # inputs[i, 21:25] = list(prog[4])
            inputs[i, 13:17] = prog[5:9]  # cur chord
            # inputs[i, 29:36] = list(prog[9])
            # inputs[i, 36] = prog[10]
            # inputs[i, 37:] = list(prog[11])
            inputs[i, 17] = prog[12]
            inputs[i, 18] = prog[15]

            # soprano[i]  = prog[12]
            alto[i]     = prog[13]
            tenor[i]    = prog[14]
            # bass[i]     = prog[15]

            count += 1
        yield (inputs, {'alto': alto, 'tenor': tenor})
    

def read_data(input_file):
    """
    Handles an entire .txt input file and returns ndarrays, splitting up
    the inputs and outputs

    Returns:
        ndarray for each inputs, alto, tenor
    """
    # get data
    dataset = []
    with open(input_file) as f:
        for line in f.read().splitlines():
            dataset.append([obj for obj in line.split()])
    f.close()

    dataset = np.array(dataset, dtype=np.int32)
    
    # split into inputs and outputs
    split = np.hsplit(dataset, [41])
    chord = split[1]

    inputs = np.zeros((dataset.shape[0], 19))
    split = np.asarray(split[0])
    inputs[:, :13] = split[:, :13]  # key sig
    inputs[:, 13:17] = split[:, 25:29]  # cur chord
    
    split = np.hsplit(chord, 4)
    soprano, alto, tenor, bass = split[0], split[1], split[2], split[3]

    inputs[:, 17] = soprano[:, 0]
    inputs[:, 18] = bass[:, 0]

    return inputs, soprano, alto, tenor, bass