"""
Project: ChordNet
Author: Gabriel Abrantes
Email: gabrantes99@gmail.com
Date: 6/29/2019
Title: satb.py
Description: 
    A class to represent and encapsulate four (4) voices:
    soprano, alto, tenor, bass (SATB)
"""

from utils.voice import Voice

class Satb():
    def __init__(self):
        self.voices = [
            Voice("C4", "G5"),  # soprano
            Voice("G3", "D5"),  # alto
            Voice("C3", "G4"),  # tenor
            Voice("E2", "C4")   # bass
            ]

    def valid_chord(self, chord: list) -> bool:
        """Determine whether the given chord is valid for the SATB choir.

        Args:
            chord: a chord of notes: [soprano, alto, tenor, bass]

        Returns:
            True if valid, False otherwise.
        """
        if len(chord) != 4:
            return False

        # check range
        for voice, note in zip(self.voices, chord):
            if not voice.in_range(note):
                return False

        # check for voice crossing
        for i in range(3):
            if chord[i] < chord[i+1]:
                return False
        return True

    def transpose_range(self, chord: list) -> tuple:
        """
        Determine the valid number of half-steps to tranpose the
        given chord up and down.

        Args:
            chord: a list of notes to tranpose: [soprano, alto, tenor, bass]

        Returns:
            Number of valid steps down (nonpositive)
            Number of valid steps up (nonnegative)
        
        Raises:
            ValueError: if chord is invalid.
        """
        if self.valid_chord(chord):
            tranpose_down = -self.voices[3].range - 1  # values outside the largest range
            tranpose_up = self.voices[3].range + 1

            for voice, note in zip(self.voices, chord):
                voice_down, voice_up = voice.tranpose_range(note)
                tranpose_up = min(tranpose_up, voice_up)
                tranpose_down = max(tranpose_down, voice_down)
            return tranpose_down, transpose_up
        else:
            raise ValueError("Chord is invalid", chord)


