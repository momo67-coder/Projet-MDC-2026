#!/usr/bin/python3

"""Fonctions de traitement d'images.
Version optimisée.

.. codeauthor:: R. Moitié
"""

import numpy as np

#from scipy.ndimage.filters import convolve


def copy(img):
    """Copie la couleur (3 valeurs) de chaque pixel.

    Parameters
    ----------
    img : numpy.array
        Image source.

    Returns
    -------
    res : numpy.array
        Image résultat
    """
    res = img.copy()
    return res