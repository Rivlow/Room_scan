"""
complete_walls.py : bouche les trous de mesure des murs d'un domaine voxélisé, à l'aide des classes du maillage ARKit.

Le LiDAR ne mesure pas tout (vitres, surfaces sombres ou brillantes, coins jamais visés) : là où aucun rayon n'est
arrivé, scan_to_domain.py laisse de l'air, et l'air de la pièce passe par ces trous dans l'épaisseur des murs.

Méthode (murs alignés sur les axes x et y par scan_to_domain.py) :
  1. les plans de murs viennent des seuls triangles ARKit de classe « mur » tournés vers x ou y : une armoire ou le
     dos d'un canapé ne font pas de plan ;
  2. dans chaque plan, la surface du mur est ce que le maillage classe mur, porte ou fenêtre, plus les mailles déjà
     solides ; les trous sont les zones qu'elle entoure complètement (le bord du domaine compte comme mur), jusqu'à
     MAX_HOLE_AREA ;
  3. une maille du plan devient solide si elle est dans un trou ou sur un triangle de mur, et qu'aucun rayon (au bruit
     près) ne l'a traversée : un passage réellement vu reste de l'air ;
  4. les portes et fenêtres d'ARKit restent telles que mesurées : fermées, les rayons s'y arrêtent (solide) ; ouvertes,
     ils les traversent (air).
"""
import numpy as np
from scipy import ndimage

from detect_openings import read_ply, splat

WALL = 1                    # classes ARKit (detect_openings.py, mesh.ply de RoomScan)
OPENINGS = (6, 7)           # fenêtre, porte
AXIS_NORMAL = 0.9           # |composante| de la normale d'un triangle tourné vers x ou vers y
MIN_WALL_AREA = 0.5         # m² de triangles de mur dans un plan pour en faire un mur
MAX_HOLE_AREA = 1.5         # m² : au-delà, le plan peut se prolonger dans une autre pièce, on ne bouche pas
STRAY_RAYS = 2              # rayons à travers une maille comptés comme bruit de profondeur (pas un passage vu)
PLANE_HALF = 1              # mailles de part et d'autre d'un plan qui appartiennent au mur


def wall_planes(profile):
    """Indices des plans de murs : maxima locaux du profil d'aire de mur d'au moins MIN_WALL_AREA."""
    n = len(profile)
    return [p for p in range(n) if profile[p] >= MIN_WALL_AREA
            and profile[p] >= profile[max(p - 1, 0)] and profile[p] >= profile[min(p + 1, n - 1)]]


def plane_holes(surface, dx):
    """Zones d'un plan entourées par la surface du mur (bord compris), de MAX_HOLE_AREA au plus."""
    holes = ndimage.binary_fill_holes(np.pad(surface, 1, constant_values=True))[1:-1, 1:-1] & ~surface
    lab, n = ndimage.label(holes)
    if n == 0:
        return holes
    area = ndimage.sum(holes, lab, np.arange(1, n + 1)) * dx * dx
    return np.isin(lab, 1 + np.flatnonzero(area <= MAX_HOLE_AREA))


def complete_walls(solid, free, mesh_path, Rz, lo, dx):
    """Masque solide complété (copie) et liste des plans de murs (axe 'x' ou 'y', position en m)."""
    V, F = read_ply(mesh_path)
    P = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1) @ Rz.T             # ARKit -> moteur, murs sur les axes
    A, B, C = (P[F["i"][:, k]] for k in range(3))
    n = np.cross(B - A, C - A)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shape = solid.shape
    opening_tri = np.flatnonzero(np.isin(F["c"], OPENINGS))
    opening_area = splat(P, F, opening_tri, lo, dx, shape) if len(opening_tri) else np.zeros(shape)

    out = solid.copy()
    planes = []
    for ax in (0, 1):
        tri = np.flatnonzero((F["c"] == WALL) & (np.abs(n[:, ax]) > AXIS_NORMAL))
        if len(tri) == 0:
            continue
        area = splat(P, F, tri, lo, dx, shape)
        profile = area.sum(axis=tuple(a for a in range(3) if a != ax))
        for p in wall_planes(profile):
            band = range(max(p - PLANE_HALF, 0), min(p + PLANE_HALF + 1, shape[ax]))
            surface = (np.take(area, band, ax) > 0).any(ax) | (np.take(opening_area, band, ax) > 0).any(ax) \
                | np.take(solid, band, ax).any(ax)
            opening = ndimage.binary_dilation((np.take(opening_area, band, ax) > 0).any(ax))
            unseen = np.take(free, p, ax) <= STRAY_RAYS
            add = (plane_holes(surface, dx) | (np.take(area, p, ax) > 0)) & unseen & ~opening
            idx = [slice(None)] * 3
            idx[ax] = p
            out[tuple(idx)] |= add
            planes.append(("xy"[ax], round(float(lo[ax] + (p + 0.5) * dx), 3)))
    return out, planes
