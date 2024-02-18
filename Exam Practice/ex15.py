import numpy as np


def color_give(x):
    '''Returns the color given a wavelength in nanometers(nm)'''
    assert x>=380, 'Not a valid color'
    assert x<750, 'Not a valid color'
    if x<450:
        return 'Violet'
    if x<495:
        return 'Blue'
    if x < 570:
        return 'Green'
    if x<590:
        return 'Yellow'
    if x <620:
        return 'Orange'
    else:
        return 'Red'
    
np
print(color_give(749))