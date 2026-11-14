"""
Recalage de deux scans (register_scans.py) puis conversion commune (scan_to_domain.py avec plusieurs scans).

La pièce synthétique de test_format.py est scannée deux fois dans deux repères ARKit différents (autre rotation
autour de la verticale, autre origine), le second scan depuis deux positions seulement (recouvrement partiel). On
vérifie que la transformation retrouvée est celle qui a servi à fabriquer les scans, que les maillages ARKit sont
fusionnés dans le repère de la référence, et que le domaine converti à partir des deux scans a toujours les
dimensions de la pièce (un mauvais recalage doublerait les murs).

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
from test_format import OFFSET, ROOM, YAW, make_scan, rot_y          # noqa: E402
from test_complete_walls import write_room_mesh                      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
YAW_B = math.radians(-70.0)
OFFSET_B = np.array([-1.5, 0.4, 3.0])
POSITIONS_B = ([1.0, 1.4, 2.4], [2.0, 1.1, 0.6])


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def scans(tmp_path_factory):
    base = tmp_path_factory.mktemp("register")
    kf_a = make_scan(base / "a")
    kf_b = make_scan(base / "b", yaw=YAW_B, offset=OFFSET_B, positions=POSITIONS_B)
    write_room_mesh(base / "a" / "mesh.ply")
    write_room_mesh(base / "b" / "mesh.ply", yaw=YAW_B, offset=OFFSET_B)
    out = run(ROOT / "register_scans.py", kf_a, kf_b, "--stride", "2")
    info = json.loads((base / "b" / "alignment.json").read_text(encoding="utf-8"))
    return base, kf_a, kf_b, info, out


def test_transform_matches_ground_truth(scans):
    base, _, _, info, out = scans
    M = np.array(info["transform_arkit"])
    # vérité : monde_A <- pièce <- monde_B
    Ra, Rb = rot_y(YAW), rot_y(YAW_B)
    R = Ra @ Rb.T
    t = OFFSET - R @ OFFSET_B
    assert np.allclose(M[:3, :3], R, atol=math.radians(1.5)), (M, R, out)
    assert np.allclose(M[:3, 3], t, atol=0.06), (M[:3, 3], t, out)
    assert info["fitness"] > 0.5, info
    assert (base / "b" / "alignment.png").exists()


def test_domain_from_both_scans(scans):
    base, kf_a, kf_b, _, _ = scans
    out = base / "out"
    run(ROOT / "scan_to_domain.py", kf_a, kf_b, "--out", out, "--dx", "0.1", "--name", "deux", "--margin", "0")
    case = json.loads((out / "deux.json").read_text(encoding="utf-8"))
    g = case["grid"]
    dims = sorted([g["Lx"], g["Ly"]])
    assert abs(dims[0] - ROOM[2]) <= 0.25 and abs(dims[1] - ROOM[0]) <= 0.25, dims
    assert len(case["scan"]["sources"]) == 2
    solids = np.load(out / "solids.npy")
    air = (solids == 0).sum() * 0.1 ** 3
    assert 25.0 < air < 31.0, air
    # maillages fusionnés dans le repère de la référence : deux fois 24 sommets, murs confondus
    from detect_openings import read_ply
    assert case["scan"]["mesh"] == "mesh_merged.ply"
    V, F = read_ply(out / "mesh_merged.ply")
    assert len(V) == 48 and len(F) == 24
    A, B = V[:24, :3], V[24:, :3]
    d = np.linalg.norm(A[:, None] - B[None], axis=-1).min(1)
    assert d.max() < 0.06, d.max()


def test_missing_alignment_is_refused(tmp_path):
    kf_a = make_scan(tmp_path / "a", positions=([1.0, 1.4, 2.4],))
    kf_b = make_scan(tmp_path / "b", positions=([1.0, 1.4, 2.4],))
    r = subprocess.run([sys.executable, str(ROOT / "scan_to_domain.py"), str(kf_a), str(kf_b), "--out", str(tmp_path / "out")],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode != 0
    assert "alignment.json" in r.stderr + r.stdout
