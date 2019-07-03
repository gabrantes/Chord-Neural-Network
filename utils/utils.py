"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/30/2019
Title: utils.py
Description: A set of helper functions.
"""

import matplotlib.pyplot as plt

def note_to_num(note_str: str) -> int:
    """Convert a musical pitch from string representation to an integer.

    Args: 
        note_str: The string representation of a musical pitch, e.g. 'C4'.

    Returns: 
        The corresponding integer of the pitch.

    Raises:
        ValueError: if note is invalid.
    """
    note_map = {
        'Cb': -1,    'C': 0,     'C#': 1,
        'Db': 1,     'D': 2,     'D#': 3,
        'Eb': 3,     'E': 4,     'E#': 5,
        'Fb': 4,     'F': 5,     'F#': 6,
        'Gb': 6,     'G': 7,     'G#': 8,
        'Ab': 8,     'A': 9,     'A#': 10,
        'Bb': 10,    'B': 11,    'B#': 12
    }
    if note_str[:-1] not in note_map:
        raise ValueError("Cannot convert invalid note to int.", note_str)
    note_int = note_map[note_str[:-1]]
    octave = int(note_str[-1])
    note_int = 12*octave + note_int
    return note_int

def num_to_note(note_int: int) -> str:
    """
    Convert a musical pitch from integer representation to a string.
    "Merge" enharmonic equivalents, e.g. the integers for 'F#' and 'Gb'
    just become the string 'F#'.

    Args: 
        note_int: The integer representation of a musical pitch.

    Returns:
        The corresponding string for the pitch.
    """
    octave = str(note_int // 12)
    rev_note_map = {
        0: 'C',      1: 'C#',    2: 'D',
        3: 'D#',     4: 'E',     5: 'F',
        6: 'F#',     7: 'G',     8: 'G#',
        9: 'A',     10: 'A#',   11: 'B'
    }
    note_str = rev_note_map[note_int % 12] + octave
    return note_str

def one_hot(val: int, length: int) -> list:
    """Returns a one-hot array of the given length, where the value
    at index=val is 1.
    """
    if val >= length:
        raise ValueError("Invalid: val >= length", val, length)
    arr = [0] * length
    arr[val] = 1
    return arr

# some code to test the functions
if __name__ == "__main__":
    key1 = 0
    key1 = num_to_note(key1)
    arr1 = [53, 50, 47, 43]
    key2 = 9
    key2 = num_to_note(key2)
    arr2 = [50, 47, 44, 40]
    arr1 = [num_to_note(el) for el in arr1]
    arr2 = [num_to_note(el) for el in arr2]
    print("before augment:\t{}\t{}".format(arr1, key1))
    print("after augment:\t{}\t{}".format(arr2, key2))

    test_note = "G4"
    print("\n" + test_note)
    print(note_to_num(test_note))
    print("reversed:")
    print(num_to_note(note_to_num(test_note)))

    print("\nCb1 == B0")
    result = str(note_to_num("Cb1")) + " == " + str(note_to_num("B0"))
    print(result)
    print("reversed:")
    result = num_to_note(note_to_num("Cb1")) + " == " + num_to_note(note_to_num("B0"))
    print(result)

    print("\nB#0 == C1")
    result = str(note_to_num("B#0")) + " == " + str(note_to_num("C1"))
    print(result)
    print("reversed:")
    result = num_to_note(note_to_num("B#0")) + " == " + num_to_note(note_to_num("C1"))
    print(result)
