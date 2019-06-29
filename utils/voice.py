"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/29/2019
Title: voice.py
Description: A class to represent a generic voice.
"""

from utils.utils import note_to_num

class Voice():
    def __init__(self, low_note: str, high_note: str):
        self.low_note = low_note
        self.high_note = high_note

        self.low_num = note_to_num(low_note)
        self.high_num = note_to_num(high_note)
        
        self.range = self.high_num - self.low_num + 1  # number of notes in voice's range

    def in_range(note: str) -> bool:
        """Determine whether the given note is in the voice's range."""
        note_num = note_to_num(note)
        return (low_num <= note_num) and (note_num <= high_num)

    def transpose_range(note: str) -> tuple:
        """Determine the valid number of half-steps to tranpose the
        given note up and down while staying within the voice's range.

        Returns:
            Number of valid steps down (nonpositive)
            Number of valid steps up (nonnegative)

        Raises:
            ValueError: if note is out of range.
        """  
        if self.in_range(note):
            tranpose_up = self.high_num - note_to_num(note)
            tranpose_down = self.low_num - note_to_num(note)
            return tranpose_down, tranpose_up
        else:
            raise ValueError('Note is out-of-range', note)