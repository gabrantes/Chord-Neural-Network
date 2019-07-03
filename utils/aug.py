from utils.satb import Satb
import numpy as np

def augment(progression: list) -> list:
    """`Augment` a progression by tranposing it to a new key.

    Args:
        progression: a list representing one line from the dataset (with scaled chords)

    Returns:
        A list representing the new, tranposed progression (still with scaled chords).
    """
    assert len(progression) == 16

    satb = Satb()
    lo, hi = satb.transpose_range(satb.unscale(progression[5:9]), satb.unscale(progression[12:16]))
    shift = np.random.randint(lo, hi+1)

    new_progression = progression[:]            

    # tonic (key signature)
    new_progression[0] = (new_progression[0] + shift) % 12

    for j in range(5, 9):  # cur chord
        new_progression[j] += shift
    for j in range(12, 16):  # next chord
        new_progression[j] += shift
    
    return new_progression