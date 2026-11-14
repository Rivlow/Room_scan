"""
Préparation du scan dans Physical_engine : l'aperçu (--preview) et la voxélisation avec un repère imposé (--yaw,
--floor, --ceiling, --crop), sur la pièce synthétique de test_format.py à laquelle on ajoute un petit maillage ARKit.

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
from test_format import make_scan   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scan_to_domain   # noqa: E402

# trois sommets ARKit (x droite, y haut, z arrière) et leur normale, un triangle de classe « porte » (7)
MESH_V = np.array([[5.0, -0.3, -2.0, 0, 1, 0], [6.0, -0.3, -2.0, 0, 1, 0], [5.0, 0.7, -2.5, 0, 0, 1]], np.float32)
DOOR = 7


def write_mesh(path):
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
            + "".join(f"property float {p}\n" for p in ("x", "y", "z", "nx", "ny", "nz"))
            + "element face 1\nproperty list uchar int vertex_indices\nproperty uchar class\nend_header\n")
    face = np.zeros(1, np.dtype([("n", "u1"), ("i", "<i4", 3), ("c", "u1")]))
    face["n"], face["i"], face["c"] = 3, [0, 1, 2], DOOR
    path.write_bytes(head.encode("ascii") + MESH_V.tobytes() + face.tobytes())


def run(*args):
    r = subprocess.run([sys.executable, str(ROOT / "scan_to_domain.py"), *map(str, args)], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    base = tmp_path_factory.mktemp("prep")
    kf = make_scan(base / "piece")
    write_mesh(base / "piece" / "mesh.ply")
    out = base / "apercu"
    run(kf, "--out", out, "--preview", "--margin", "0")
    return base, kf, json.loads((out / "preview.json").read_text(encoding="utf-8")), out


def test_preview_frame(scan):
    _, _, info, out = scan
    assert not (out / "solids.npy").exists()                # l'aperçu ne voxélise pas
    yaw = info["yaw_deg"] % 90
    assert min(abs(yaw - 25), abs(yaw - 115), abs(yaw + 65)) < 2.0, yaw
    assert info["ceiling_scanned"] and abs(info["ceiling"] - info["floor"] - 2.5) < 0.06
    x0, x1, y0, y1 = info["crop"]
    assert sorted([x1 - x0, y1 - y0]) == pytest.approx([3.0, 4.0], abs=0.25)


def test_preview_confidence(scan):
    _, _, info, _ = scan
    n0, n1, n2 = info["confidence"]                         # la pièce synthétique est toute en confiance haute
    assert n0 == 0 and n1 == 0 and n2 == 192 * 256 * info["frames"]


def test_preview_mesh_in_the_rotated_frame(scan):
    _, _, info, out = scan
    raw = (out / info["mesh"]).read_bytes()
    assert raw[:4] == b"PMSH"
    version, nv, nt = np.frombuffer(raw, "<u4", 3, 4)
    assert (version, nv, nt) == (1, 3, 1)
    V = np.frombuffer(raw, "<f4", nv * 6, 16).reshape(nv, 6)
    F = np.frombuffer(raw, "<u4", nt * 3, 16 + nv * 24)
    C = np.frombuffer(raw, np.uint8, nt, 16 + nv * 24 + nt * 12)
    assert list(F) == [0, 1, 2] and C[0] == DOOR
    yaw = math.radians(info["yaw_deg"])
    c, s = math.cos(-yaw), math.sin(-yaw)
    for v, w in zip(MESH_V, V):
        e = np.array([v[0], -v[2], v[1]])                   # ARKit -> moteur, puis rotation des murs
        assert w[:3] == pytest.approx([c * e[0] - s * e[1], s * e[0] + c * e[1], e[2]], abs=1e-4)
    assert V[2, 3:] == pytest.approx([c * 0 - s * -1, s * 0 + c * -1, 0], abs=1e-5)       # normale tournée aussi


def test_imposed_frame(scan):
    base, kf, info, _ = scan
    x0, x1, y0, y1 = info["crop"]
    crop = [x0 + 0.5, x1 - 0.5, y0 + 0.3, y1 - 0.3]          # la pièce coupée de 50 cm et 30 cm de chaque côté
    floor, ceiling = info["floor"], info["floor"] + 2.0      # plafond imposé sous le vrai
    out = base / "impose"
    run(kf, "--out", out, "--dx", "0.1", "--name", "piece", "--yaw", info["yaw_deg"], "--floor", floor,
        "--ceiling", ceiling, "--crop", *crop)
    case = json.loads((out / "piece.json").read_text(encoding="utf-8"))
    g, sc = case["grid"], case["scan"]
    assert g["nx"] == math.ceil((crop[1] - crop[0]) / 0.1) and g["ny"] == math.ceil((crop[3] - crop[2]) / 0.1)
    assert g["nz"] == 20 + 2                                 # 2 m de pièce, une dalle dessous et une dessus
    assert sc["yaw_deg"] == pytest.approx(info["yaw_deg"], abs=1e-3)
    assert sc["origin_arkit_floor"] == pytest.approx([crop[0], crop[2], floor], abs=1e-4)
    assert sc["ceiling_height"] == pytest.approx(2.0)
    assert "imposé" in (out / "rapport.txt").read_text(encoding="utf-8")


def test_confidence_rank_spread_over_bytes():
    """Polycam étale la confiance sur 0..255 (0 / 54 / 255) ; RoomScan l'écrit en 0 / 1 / 2."""
    assert list(scan_to_domain.confidence_rank(np.array([0, 54, 255], np.uint8))) == [0, 1, 2]
    assert list(scan_to_domain.confidence_rank(np.array([0, 1, 2], np.uint8))) == [0, 1, 2]
