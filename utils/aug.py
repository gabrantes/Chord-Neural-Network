"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 7/2/2019
Filename: aug.py
Description: 
    Contains functions for augmenting the dataeset.
"""

from satb.satb import Satb
from utils.utils import one_hot
import numpy as np

def augment(progression: list) -> list:
    """
    `Augment` a progression by tranposing it to a new key.

    Args:
        progression: a list representing one line from the dataset 
                    (with scaled chords and one-hot tuples)

    Returns:
        A list representing the new, tranposed progression
        (still with scaled chords and one-hot tuples).
    """
    assert len(progression) == 16

    satb = Satb()
    lo, hi = satb.transpose_range(satb.unscale(progression[5:9]), satb.unscale(progression[12:16]))
    shift = np.random.randint(lo, hi+1)

    new_progression = progression[:]            

    # tonic (key signature)
    cur_key = new_progression[0].index(1)
    new_key = (cur_key + shift) % 12
    new_progression[0] = tuple(one_hot(new_key, 12))

    for j in range(5, 9):  # cur chord
        new_progression[j] += shift
    for j in range(12, 16):  # next chord
        new_progression[j] += shift
    
    return new_progression