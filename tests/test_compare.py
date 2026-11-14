"""
Comparaison de deux versions d'un domaine (compare_scans.py) : la pièce synthétique, puis la même avec une armoire
en plus (même repère ARKit, comme un scan repris). L'armoire doit apparaître comme « apparue », rien ne doit avoir
disparu, et la grille du nouveau domaine, détectée indépendamment, doit être rééchantillonnée sans erreur.

    python -m pytest tests -q        (depuis Room_scan/)
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_format import OFFSET, YAW, make_scan, rot_y, slab_hit      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
# buffet collé au mur x = 4, posé au sol, 1 m de haut (repère pièce). Seules ses faces vues (devant, côtés, dessus)
# deviennent solides : les murs de la pièce synthétique sont la frontière de la grille, pas des mailles solides, donc
# l'intérieur du buffet n'est pas une poche close et reste de l'air (--unknown air). La boîte englobante de ce qui
# est « apparu » est néanmoins celle du buffet.
CAB_LO, CAB_HI = np.array([3.3, 0.0, 0.3]), np.array([4.0, 1.0, 1.1])


def cabinet_box_domain(case):
    """Boîte englobante du buffet dans le repère du domaine de référence (pièce tournée et décalée dans le monde ARKit,
    puis repère du moteur x, -z, y, puis rotation et origine détectées par scan_to_domain)."""
    import math
    corners = np.array([[x, y, z] for x in (CAB_LO[0], CAB_HI[0]) for y in (CAB_LO[1], CAB_HI[1]) for z in (CAB_LO[2], CAB_HI[2])])
    ark = corners @ rot_y(YAW).T + OFFSET
    eng = np.stack([ark[:, 0], -ark[:, 2], ark[:, 1]], 1)
    sc = case["scan"]
    yaw = math.radians(sc["yaw_deg"])
    c, s = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o = sc["origin_arkit_floor"]
    lo = np.array([o[0], o[1], o[2] - sc["dx"]])
    q = eng @ Rz.T - lo
    return q.min(0), q.max(0)
DX = 0.1


def cabinet(o, dn, t):
    return np.minimum(t, slab_hit(o, dn, CAB_LO, CAB_HI, False))


def convert(kf, out, name):
    r = subprocess.run([sys.executable, str(ROOT / "scan_to_domain.py"), str(kf), "--out", str(out), "--dx", str(DX),
                        "--name", name, "--margin", "0"], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return out / f"{name}.json"


@pytest.fixture(scope="module")
def diff(tmp_path_factory):
    base = tmp_path_factory.mktemp("compare")
    ref = convert(make_scan(base / "v1"), base / "d1", "v1")
    new = convert(make_scan(base / "v2", edit=cabinet), base / "d2", "v2")
    r = subprocess.run([sys.executable, str(ROOT / "compare_scans.py"), str(ref), str(new), "--out", str(base / "diff")],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads((base / "diff" / "diff.json").read_text(encoding="utf-8"))
    info["ref_case"] = json.loads(ref.read_text(encoding="utf-8"))
    return base, info, np.load(base / "diff" / "diff.npy"), r.stdout


def test_cabinet_appears(diff):
    base, info, d, out = diff
    vol = float(np.prod(CAB_HI - CAB_LO))                      # 0,7 x 1,0 x 0,8 = 0,56 m3 (coque vue : ~0,15 m3)
    assert 0.1 * vol < info["added_m3"] <= 1.0 * vol, (info["added_m3"], vol, out)
    assert info["removed_m3"] < 0.05, (info["removed_m3"], out)
    assert info["compared_share"] > 0.9, info["compared_share"]
    assert (base / "diff" / "apercu_diff.png").exists() and (base / "diff" / "rapport_diff.txt").exists()


def test_cabinet_location(diff):
    _, info, _, out = diff
    assert len(info["added_zones"]) >= 1, out
    zone = info["added_zones"][0]
    lo, hi = cabinet_box_domain(info["ref_case"])
    # la zone apparue est le buffet, à une maille près (0,1 m) sur chaque face, dans le repère du domaine de référence
    assert np.all(np.abs(np.array(zone["box_min"]) - lo) < 0.2), (zone, lo, hi, out)
    assert np.all(np.abs(np.array(zone["box_max"]) - hi) < 0.2), (zone, lo, hi, out)
    assert zone["volume_m3"] > 0.05, zone
    # tout ce qui est apparu est dans cette zone, à peu de chose près
    assert zone["volume_m3"] > 0.8 * info["added_m3"], (zone, info["added_m3"])


def test_codes(diff):
    _, _, d, _ = diff
    assert d.dtype == np.uint8 and set(np.unique(d)) <= {0, 1, 2, 3, 4}
    assert (d[:, :, 0] == 1).all() and (d[:, :, -1] == 1).all()
