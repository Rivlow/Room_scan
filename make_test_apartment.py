"""
Masque synthétique d'appartement pour les tests de Physical_engine (assets/cases/appartement_solids.npy).

Grille 60 x 40 x 28 mailles de 10 cm : 6 x 4 m au sol, 2,6 m sous plafond, une dalle solide sous le sol (k = 0) et une
au-dessus du plafond (k = 27). Murs extérieurs d'une maille ; deux pièces séparées par une cloison à x = 3,5 m avec une
porte de 0,9 x 2,0 m ; une table (plateau à 0,7-0,8 m sur quatre pieds) ; une étagère murale (surplomb) ; une fenêtre
vitrée (valeur 2) dans le mur sud de la seconde pièce, entre 0,9 et 2,1 m.

    python make_test_apartment.py <fichier .npy>
"""
import sys

import numpy as np

nx, ny, nz = 60, 40, 28
s = np.zeros((nx, ny, nz), np.uint8)
s[:, :, 0] = 1                       # sous le sol
s[:, :, nz - 1] = 1                  # au-dessus du plafond
s[0, :, :] = s[nx - 1, :, :] = 1     # murs extérieurs
s[:, 0, :] = s[:, ny - 1, :] = 1
s[35, :, :] = 1                      # cloison
s[35, 18:27, 1:21] = 0               # porte 0,9 x 2,0 m
s[8:20, 12:22, 8] = 1                # plateau de table à 0,7-0,8 m
for i, j in ((8, 12), (19, 12), (8, 21), (19, 21)):
    s[i, j, 1:8] = 1                 # pieds
s[1:4, 5:30, 16] = 1                 # étagère murale à 1,5 m, 30 cm de profondeur
s[42:54, 0, 10:22] = 2               # fenêtre vitrée, mur sud de la seconde pièce
np.save(sys.argv[1], s)
print(sys.argv[1], s.shape, "solide", int((s == 1).sum()), "vitre", int((s == 2).sum()), "air", int((s == 0).sum()))
