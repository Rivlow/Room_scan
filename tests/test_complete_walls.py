"""
Complétion des murs (complete_walls.py, scan_to_domain.py --complete-walls) sur la pièce synthétique de test_format.py,
percée de deux façons :
  - un trou de mesure dans le mur x = 4 m (aucune profondeur rendue, comme une vitre ou une surface noire) : sans
    complétion la paroi reste ouverte, avec complétion elle est bouchée ;
  - une alcôve réelle derrière le mur z = 0 (les rayons la traversent) : elle doit rester de l'air, même si le
    maillage ARKit classe tout le plan comme mur.

    python -m pytest tests -q        (depuis Room_scan/)
"""
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_format import ROOM, YAW, make_scan, rot_y, slab_hit   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OFFSET = np.array([5.0, -0.3, -2.0])                      # pièce -> monde, comme synthetic_frames
HOLE = dict(y=(0.9, 1.7), z=(1.0, 2.0))                    # sur le mur x = 4 (repère de la pièce, y vers le haut)
ALCOVE_LO, ALCOVE_HI = np.array([1.5, 0.0, -1.0]), np.array([2.5, 2.0, 0.0])
DX = 0.1


def pierce(o, dn, t):
    """Trou de mesure sur le mur x = 4, alcôve derrière le mur z = 0."""
    p = o + dn * t[..., None]
    hole = (np.abs(p[..., 0] - ROOM[0]) < 1e-6) & (p[..., 1] > HOLE["y"][0]) & (p[..., 1] < HOLE["y"][1]) \
        & (p[..., 2] > HOLE["z"][0]) & (p[..., 2] < HOLE["z"][1])
    door = (np.abs(p[..., 2]) < 1e-6) & (p[..., 0] > ALCOVE_LO[0]) & (p[..., 0] < ALCOVE_HI[0]) & (p[..., 1] < ALCOVE_HI[1])
    t = np.where(door, slab_hit(o, dn, ALCOVE_LO, ALCOVE_HI, True), t)
    return np.where(hole, np.inf, t)


def write_room_mesh(path, yaw=YAW, offset=OFFSET):
    """Maillage ARKit de la pièce : les 4 murs entiers en classe « mur » (y compris devant l'alcôve), sol, plafond ;
    yaw et offset la placent dans le monde comme make_scan."""
    X, Y, Z = ROOM
    quads = [([0, 0, 0], [0, Y, 0], [0, Y, Z], [0, 0, Z], 1), ([X, 0, 0], [X, Y, 0], [X, Y, Z], [X, 0, Z], 1),
             ([0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0], 1), ([0, 0, Z], [X, 0, Z], [X, Y, Z], [0, Y, Z], 1),
             ([0, 0, 0], [X, 0, 0], [X, 0, Z], [0, 0, Z], 2), ([0, Y, 0], [X, Y, 0], [X, Y, Z], [0, Y, Z], 3)]
    Rw = rot_y(yaw)
    verts, faces = [], []
    for *corners, cls in quads:
        base = len(verts)
        verts += [list(Rw @ np.array(c, float) + np.asarray(offset, float)) + [0.0, 0.0, 0.0] for c in corners]
        faces += [(base, base + 1, base + 2, cls), (base, base + 2, base + 3, cls)]
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
            + "".join(f"property float {p}\n" for p in ("x", "y", "z", "nx", "ny", "nz"))
            + f"element face {len(faces)}\nproperty list uchar int vertex_indices\nproperty uchar class\nend_header\n")
    F = np.zeros(len(faces), np.dtype([("n", "u1"), ("i", "<i4", 3), ("c", "u1")]))
    F["n"] = 3
    F["i"] = [f[:3] for f in faces]
    F["c"] = [f[3] for f in faces]
    path.write_bytes(head.encode("ascii") + np.array(verts, "<f4").tobytes() + F.tobytes())


def voxelize(kf, out, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "scan_to_domain.py"), str(kf), "--out", str(out), "--dx", str(DX),
                        "--name", "piece", "--margin", "0", *extra], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    case = json.loads((out / "piece.json").read_text(encoding="utf-8"))
    return case, np.load(out / "solids.npy").astype(bool), (out / "rapport.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def domains(tmp_path_factory):
    base = tmp_path_factory.mktemp("murs")
    kf = make_scan(base / "piece", pierce)
    write_room_mesh(base / "piece" / "mesh.ply")
    return voxelize(kf, base / "brut"), voxelize(kf, base / "complet", "--complete-walls")


def cell(case, solid, room_point):
    """Maille du domaine qui contient un point du repère de la pièce (None hors du domaine)."""
    w = rot_y(YAW) @ np.array(room_point, float) + OFFSET
    e = np.array([w[0], -w[2], w[1]])                     # ARKit -> moteur
    yaw = math.radians(case["scan"]["yaw_deg"])
    c, s = math.cos(-yaw), math.sin(-yaw)
    r = np.array([c * e[0] - s * e[1], s * e[0] + c * e[1], e[2]])
    o = case["scan"]["origin_arkit_floor"]
    ijk = np.floor((r - np.array([o[0], o[1], o[2] - DX])) / DX).astype(int)
    return tuple(ijk) if np.all((ijk >= 0) & (ijk < solid.shape)) else None


def wall_closed_at(case, solid, y, z):
    """Le mur x = 4 est fermé au point (y, z) : une maille solide sur la traversée du mur (3,85 à 4,15 m)."""
    cells = [cell(case, solid, (x, y, z)) for x in np.arange(3.85, 4.16, 0.05)]
    return any(solid[c] for c in cells if c is not None)


def test_measurement_hole_is_filled(domains):
    (case0, solid0, _), (case1, solid1, report) = domains
    centre = (sum(HOLE["y"]) / 2, sum(HOLE["z"]) / 2)
    assert not wall_closed_at(case0, solid0, *centre)      # sans complétion : le trou est de l'air
    assert wall_closed_at(case1, solid1, *centre)
    assert "mailles bouchées" in report


def test_seen_alcove_stays_open(domains):
    for case, solid, _ in domains:
        assert not solid[cell(case, solid, (2.0, 1.0, -0.5))]       # dans l'alcôve
        assert not solid[cell(case, solid, (2.0, 1.0, 0.0))]        # dans son ouverture, sur le plan du mur


def room_point(case, ijk):
    """Centre d'une maille du domaine -> point du repère de la pièce (inverse de cell)."""
    o = case["scan"]["origin_arkit_floor"]
    r = np.array([o[0], o[1], o[2] - DX]) + (np.array(ijk) + 0.5) * DX
    yaw = math.radians(case["scan"]["yaw_deg"])
    c, s = math.cos(yaw), math.sin(yaw)
    e = np.array([c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]])
    w = np.array([e[0], e[2], -e[1]])                     # moteur -> ARKit
    return rot_y(YAW).T @ (w - OFFSET)


def test_only_wall_planes_change(domains):
    """La complétion n'enlève jamais de solide, et n'en ajoute que sur les murs (à une maille près), jamais dans la pièce."""
    (_, solid0, _), (case1, solid1, _) = domains
    assert not (solid0 & ~solid1).any()
    added = np.argwhere(solid1 & ~solid0)
    assert len(added) > 0
    for ijk in added:
        p = room_point(case1, ijk)
        to_wall = min(abs(p[0]), abs(p[0] - ROOM[0]), abs(p[2]), abs(p[2] - ROOM[2]))
        assert to_wall <= 1.5 * DX, (ijk, p)
