#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Fonctions de traitement d'images.
Version non optimisée.

.. codeauthor:: R. Moitié
"""

import numpy as np


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
    res = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                res[i, j, k] = img[i, j, k]
    return res