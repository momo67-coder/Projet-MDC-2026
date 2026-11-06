# -*- coding: utf-8 -*-
"""
dessin_process.py
=================

Fonctions de traitement des dessins.

Ce module joue le même rôle que ``image_process.py`` dans le TP2 :
il contient uniquement les traitements sur les tableaux NumPy et ne dépend
pas de l'interface graphique Qt.

Convention des bitmaps
----------------------
0 = fond noir
1 = trait blanc

Les images en niveaux de gris manipulées ici sont des tableaux NumPy 2D.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =====================================================================
# PRIMITIVES DE DESSIN SUR TABLEAUX NUMPY
# =====================================================================

def paint_disk(bitmap, x, y, radius):
    """
    Peint un disque blanc centré en (x, y) dans bitmap.

    bitmap : tableau NumPy 2D
    x, y   : coordonnées du centre du disque
             x = colonne, y = ligne
    radius : rayon du disque
    """
    arr = np.asarray(bitmap)

    if arr.ndim != 2:
        raise ValueError("bitmap doit être un tableau 2D.")

    r = int(round(y))
    c = int(round(x))
    radius = max(float(radius), 0.0)

    # Le disque ne rencontre pas l'image.
    if r + radius < 0 or r - radius >= arr.shape[0]:
        return bitmap
    if c + radius < 0 or c - radius >= arr.shape[1]:
        return bitmap

    # On ne construit le masque que dans la boîte entourant le disque.
    r0 = max(0, int(np.floor(r - radius)))
    r1 = min(arr.shape[0], int(np.ceil(r + radius)) + 1)
    c0 = max(0, int(np.floor(c - radius)))
    c1 = min(arr.shape[1], int(np.ceil(c + radius)) + 1)

    # 1 & 2. Construire les grilles de coordonnées et le masque booléen
    yy, xx = np.ogrid[r0:r1, c0:c1]
    mask = (yy - y)**2 + (xx - x)**2 <= radius**2

    # 3. Mettre les pixels correspondants à 1 (trait blanc)
    bitmap[r0:r1, c0:c1][mask] = 1

    return bitmap

def paint_segment(bitmap, p0, p1, radius):
    """
    Relie les points p0 et p1 par un trait continu.

    p0 = (x0, y0)
    p1 = (x1, y1)
    """

    x0, y0 = p0
    x1, y1 = p1

    # 1. Calculer la distance entre p0 et p1.
    distance = np.hypot(x1 - x0, y1 - y0)

    # 2. Choisir la distance maximale entre deux centres de disques successifs.
    # On prend par exemple le rayon (ou 1 pixel minimum pour éviter les divisions par zéro)
    step = max(1.0, float(radius))

    # 3. Déterminer le nombre de points intermédiaires en utilisant ceil.
    n = max(2, int(np.ceil(distance / step)))

    # 4. Construire les coordonnées interpolées avec np.linspace.
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)

    # 5. Dessiner un disque en chacun de ces points.
    for x, y in zip(xs, ys):
        paint_disk(bitmap, x, y, radius)

    return bitmap

## Pour tester vos fonctions
def test_paint_disk():

    bitmap = np.zeros((28, 28), dtype=float)

    paint_disk(
        bitmap,
        x=14,
        y=14,
        radius=2
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(
        bitmap,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title("Test de paint_disk")
    plt.axis("off")
    plt.show()


def test_paint_segment():

    bitmap = np.zeros((28, 28), dtype=float)

    paint_segment(
        bitmap,
        p0=(1, 20),
        p1=(7, 15),
        radius=2
    )

    plt.figure(figsize=(4, 4))
    plt.imshow(
        bitmap,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title("Test de paint_segment")
    plt.axis("off")
    plt.show()

    
# =====================================================================
# RECENTRAGE ET CONVERSION DES DONNÉES QUICKDRAW
# =====================================================================

def compute_centering_parameters(width, height, size=28, margin=2):
    """
    Calcule les parametres de redimensionnement et de recentrage.

    Retour :
        scale : facteur d'echelle
        offx  : decalage horizontal
        offy  : decalage vertical
    """

    # Eviter les divisions par zero

    # Longueur disponible entre les deux marges.
    # Attention : les pixels sont indexes de 0 a size-1.
    usable = size - 1 - 2 * margin

    if usable <= 0:
        raise ValueError("Marge trop grande.")

    # Choisir scale pour que :
    # max(width, height) * scale = usable
    scale = usable / max(width, height)

    # Dimensions du dessin apres redimensionnement
    scaled_width = width * scale
    scaled_height = height * scale

    # Decalage = marge + moitie de l'espace libre
    offx = margin + (usable - scaled_width) / 2
    offy = margin + (usable - scaled_height) / 2

    return scale, offx, offy



def image_converter(strokes, size=28, margin=2, radius=1.0):
    """
    Convertit un dessin vectoriel QuickDraw en bitmap centré ``size x size``.

    Un dessin QuickDraw est une liste de traits. Chaque trait contient
    une liste de coordonnées x et une liste de coordonnées y.

    Exemple
    -------
    stroke = [
        [10, 15, 21, 27],   # coordonnées x
        [30, 25, 22, 20]    # coordonnées y
    ]

    Cela représente la ligne brisée
    P1=(10,30) -> P2=(15,25) -> P3=(21,22) -> P4=(27,20).

    Retour
    ------
    numpy.ndarray de forme (size, size), avec :
        0   = fond noir,
        255 = trait blanc.
    """

    # -------------------------------------------------------------
    # 1. Rassembler tous les points du dessin
    # -------------------------------------------------------------
    pts = []
    for stroke in strokes:
        xs, ys = stroke[0], stroke[1]
        pts.extend(zip(xs, ys))

    if not pts:
        return np.zeros((size, size), dtype=np.uint8)

    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    # -------------------------------------------------------------
    # 2. Calculer la boîte englobante du dessin
    # -------------------------------------------------------------
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)

    # -------------------------------------------------------------
    # 3. REDIMENSIONNER ET RECENTRER
    # -------------------------------------------------------------
    # On réutilisera cette fonction ensuite dans prepare_for_model(...)
    # pour traiter un dessin réalisé à la souris ou chargé depuis un PNG.
    scale, offx, offy = compute_centering_parameters(
        width, height, size=size, margin=margin
    )

    bitmap = np.zeros((size, size), dtype=float)
    # La transformation appliquée à chaque point est :
    #
    # x_final = offx + (x - xmin) * scale
    # y_final = offy + (y - ymin) * scale
    #
    # (x - xmin, y - ymin) ramène d'abord la boîte englobante à l'origine ;
    # scale change sa taille ;
    # offx et offy la placent au centre de l'image.

    # -------------------------------------------------------------
    # 4. Dessiner les traits dans l'image finale
    # -------------------------------------------------------------
    for stroke in strokes:
        sx, sy = stroke[0], stroke[1]

        points = [
            (
                int(round(offx + (x - xmin) * scale)),
                int(round(offy + (y - ymin) * scale))
            )
            for x, y in zip(sx, sy)
        ]

        if len(points) >= 2:
            # Trait blanc 
            for p0, p1 in zip(points[:-1], points[1:]):
                paint_segment(bitmap, p0, p1, radius)

        # Sinon on crée un petit disque autour de (x,y)
        elif len(points) == 1:
            # petit disque
            x, y = points[0]
            paint_disk(bitmap, x, y, radius)
         
    return np.uint8(np.clip(bitmap, 0.0, 1.0) * 255)

    
# =====================================================================
# LECTURE / SAUVEGARDE DES DESSINS
# =====================================================================

def load_png_as_bitmap(path):
    """
    Charge une image en niveaux de gris dans un tableau 2D dans [0, 1].

    Si le fond est majoritairement blanc, l'image est inversée afin de
    respecter la convention du projet : fond noir / trait blanc.

    Cette fonction ne recentre pas le dessin.
    """
    image = Image.open(Path(path)).convert("L")

    arr = np.asarray(image, dtype=float) / 255.0

    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr


def save_bitmap(bitmap, path):
    """
    Sauvegarde un bitmap 2D dans [0, 1] au format PNG.
    """
    arr = np.asarray(bitmap, dtype=float)

    if arr.ndim != 2:
        raise ValueError("bitmap doit être un tableau 2D.")

    arr8 = np.uint8(np.clip(arr, 0.0, 1.0) * 255)

    image = Image.fromarray(arr8, mode="L")