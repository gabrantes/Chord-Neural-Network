"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/27/2019
Title: utils.py
Description: A set of helper functions.
"""

def note_to_num(str_in):
    """
    Converts a musical pitch from string representation to an integer.

    Args: str_in -- The string representation of a musical pitch, e.g. 'C4'.

    Output: The corresponding integer of the pitch.
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
    for i in range(len(str_in)):
        if str_in[i].isdigit():
            break
    if str_in[:i] not in note_map:
        print("\nSkipped invalid note: %s"%str_in)
        return
    note_num = note_map[str_in[:i]]
    octave = int(str_in[i:])
    note_num = 12*octave + note_num
    return note_num

def num_to_note(int_in):
    """
    Converts a musical pitch from integer representation to a string.
    "Merges" enharmonic equivalents, e.g. the integers for 'F#' and 'Gb'
    just become the string 'F#'.

    Args: int_in -- The integer representation of a musical pitch.

    Output: The corresponding string for the pitch.
    """
    octave = str(int_in // 12)
    rev_note_map = {
        0: 'C',      1: 'C#',    2: 'D',
        3: 'D#',     4: 'E',     5: 'F',
        6: 'F#',     7: 'G',     8: 'G#',
        9: 'A',     10: 'A#',   11: 'B'
    }
    note = rev_note_map[int_in % 12] + octave
    return note

def SATB_dicts():
    """
    Creates dictionaries representing soprano, alto, tenor, bass (SATB).
    Tracks high notes, low notes, and range, as described in Kostka and
    Payne's 'Tonal Harmony 8th Edition' (Example 5-12 on pg 72).

    Output: The SATB dictionaries as a list: [soprano, alto, tenor, bass]
    """
    soprano = {"name": "soprano", "high_note": "G5", "low_note": "C4"}
    alto = {"name": "alto", "high_note": "D5", "low_note": "G3"}
    tenor = {"name": "tenor", "high_note": "G4", "low_note": "C3"}
    bass = {"name": "bass", "high_note": "C4", "low_note": "E2"}

    voices = [soprano, alto, tenor, bass]
    for voice in voices:
        high_num = note_to_num(voice["high_note"])
        low_num = note_to_num(voice["low_note"])
        range_ = high_num - low_num

        voice["high_num"] = high_num
        voice["low_num"] = low_num
        voice["range"] = range_

    return voices

# some code to test the functions
if __name__ == "__main__":
    soprano, alto, tenor, bass = SATB_dicts()

    for key, val in soprano.items():
        print(key, val)

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
