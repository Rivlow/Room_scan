"""
Test du format RoomScan -> scan_to_domain.py sur une pièce synthétique.

On fabrique un dossier keyframes exactement comme l'écrit l'application iPhone (JPEG 1024x768, profondeur PNG 16 bits
en mm 256x192, confiance PNG 8 bits 0/1/2, JSON de pose t_rc / fx fy cx cy) pour une pièce rectangulaire de
4 x 3 x 2,5 m tournée de 25° autour de la verticale, contenant une table (plateau mince à 0,72-0,76 m, air dessous).
La profondeur est calculée par lancer de rayons analytique. Puis on vérifie ce que retrouve le convertisseur.

    python -m pytest tests -q        (depuis Room_scan/)
"""
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROOM = np.array([4.0, 2.5, 3.0])                          # x, y (haut), z en repère ARKit, coin en (0, 0, 0)
TABLE_LO = np.array([1.2, 0.72, 1.0])
TABLE_HI = np.array([2.4, 0.76, 1.8])
YAW = math.radians(25.0)                                  # rotation de la pièce dans le repère du monde
W, H = 256, 192


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def slab_hit(o, d, lo, hi, inside):
    """Distance du rayon (o, d) à une boîte : sortie si inside, entrée sinon ; inf si pas d'intersection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o) / d
        t2 = (hi - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    if inside:
        return tmax
    hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
    return np.where(hit, tmin, np.inf)


FX = 211.0                                                # focale pour 256x192 (≈ 845 pour 1024x768)
CX, CY = W / 2, H / 2


OFFSET = np.array([5.0, -0.3, -2.0])                      # pièce -> monde : décalage (après rotation de YAW)
POSITIONS = ([1.0, 1.4, 2.4], [3.0, 1.4, 2.2], [2.0, 1.1, 0.6], [2.0, 1.6, 1.5])     # caméra, repère de la pièce


def synthetic_frames(edit=None, yaw=YAW, offset=OFFSET, positions=POSITIONS):
    """Prises de vue de la pièce synthétique : (profondeur le long de -z caméra en m, rotation caméra -> monde,
    position de la caméra dans le monde). Lancer de rayons analytique ; edit(o, dn, t) -> t modifie les distances
    (repère de la pièce) pour percer la pièce, inf = pas de mesure. yaw et offset placent la pièce dans le monde
    (un autre scan du même lieu a un autre repère ARKit)."""
    Rw = rot_y(yaw)                                       # pièce -> monde
    v, u = np.mgrid[0:H, 0:W].astype(float)
    dirs_cam = np.stack([(u - CX) / FX, -(v - CY) / FX, -np.ones_like(u)], -1)    # caméra ARKit : vers -z
    for pos in positions:
        for cam_yaw in np.radians(np.arange(0, 360, 30)):
            for pitch in np.radians([-50, -10, 35, 70]):
                Rc = rot_y(cam_yaw) @ rot_x(pitch)                     # caméra -> pièce
                o = np.array(pos)
                d = dirs_cam @ Rc.T
                dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
                t = np.minimum(slab_hit(o, dn, np.zeros(3), ROOM, True), slab_hit(o, dn, TABLE_LO, TABLE_HI, False))
                if edit is not None:
                    t = edit(o, dn, t)
                zdepth = t / np.linalg.norm(d, axis=-1)                # profondeur le long de -z caméra (d.z = -1)
                yield zdepth, Rw @ Rc, Rw @ o + np.asarray(offset, float)     # pièce décalée dans le monde


def make_scan(dst: pathlib.Path, edit=None, yaw=YAW, offset=OFFSET, positions=POSITIONS):
    kf = dst / "keyframes"
    for sub in ("images", "depth", "confidence", "cameras"):
        (kf / sub).mkdir(parents=True)
    fx = fy = FX
    cx, cy = CX, CY
    ts = 1000
    for zdepth, Rwc, tw in synthetic_frames(edit, yaw, offset, positions):
        mm = np.where(np.isfinite(zdepth), np.clip(np.round(zdepth * 1000), 0, 65535), 0).astype(np.uint16)
        name = str(ts)
        ts += 1
        Image.fromarray(mm).save(kf / "depth" / f"{name}.png")
        Image.fromarray(np.full((H, W), 2, np.uint8)).save(kf / "confidence" / f"{name}.png")
        Image.fromarray(np.full((768, 1024, 3), 128, np.uint8)).save(kf / "images" / f"{name}.jpg")
        cam = {f"t_{r}{c}": float(Rwc[r, c]) for r in range(3) for c in range(3)}
        cam.update({f"t_{r}3": float(tw[r]) for r in range(3)})
        cam.update({"fx": fx * 4, "fy": fy * 4, "cx": cx * 4, "cy": cy * 4, "width": 1024, "height": 768,
                    "timestamp": ts, "tracking_state": "Normal", "blur_score": 300.0,
                    "center_depth": float(zdepth[H // 2, W // 2])})
        (kf / "cameras" / f"{name}.json").write_text(json.dumps(cam))
    (dst / "scan.json").write_text(json.dumps({"name": "synthetique", "floorHeight": float(offset[1]),
                                               "ceilingHeight": float(offset[1] + ROOM[1])}))
    return kf


@pytest.fixture(scope="module")
def domain(tmp_path_factory):
    base = tmp_path_factory.mktemp("scan")
    kf = make_scan(base / "piece")
    out = base / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scan_to_domain.py"), str(kf), "--out", str(out), "--dx", "0.1",
                        "--name", "piece", "--margin", "0"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    case = json.loads((out / "piece.json").read_text(encoding="utf-8"))
    solids = np.load(out / "solids.npy")
    return case, solids, r.stdout


def test_files_and_shape(domain):
    case, solids, _ = domain
    g = case["grid"]
    assert solids.dtype == np.uint8
    assert solids.shape == (g["nx"], g["ny"], g["nz"])
    assert case["solids"]["file"] == "solids.npy"
    assert set(np.unique(solids)) <= {0, 1}


def test_walls_aligned_and_dimensions(domain):
    case, _, _ = domain
    g = case["grid"]
    # murs alignés sur les axes : la boîte suit la pièce (4 x 3 m, dans un ordre ou l'autre) à 2 mailles près
    dims = sorted([g["Lx"], g["Ly"]])
    assert abs(dims[0] - 3.0) <= 0.25 and abs(dims[1] - 4.0) <= 0.25, dims
    yaw = case["scan"]["yaw_deg"] % 90
    assert min(abs(yaw - 25), abs(yaw - 25 - 90), abs(yaw - 25 + 90)) < 2.0, yaw


def test_ceiling_found(domain):
    case, _, _ = domain
    assert case["scan"]["ceiling_scanned"]
    assert abs(case["scan"]["ceiling_height"] - 2.5) < 0.06


def test_table_is_an_overhang(domain):
    """Le plateau est solide et l'air passe dessous (le cas que le relief par colonne ne sait pas décrire)."""
    case, solids, _ = domain
    nz = solids.shape[2]
    k_top = 1 + int(0.74 / 0.1)                         # plateau : une dalle sous le sol (k = 0), puis 0,74 m
    columns_with_overhang = 0
    for i in range(solids.shape[0]):
        for j in range(solids.shape[1]):
            col = solids[i, j]
            if col[k_top] and not col[2:k_top - 1].any():
                columns_with_overhang += 1
    # plateau 1,2 x 0,8 m = ~96 colonnes de 10 cm ; on en exige une bonne part
    assert columns_with_overhang >= 40, columns_with_overhang
    # le volume d'air est celui de la pièce, aux murs près
    air = (solids == 0).sum() * 0.1 ** 3
    assert 25.0 < air < 31.0, air
    assert solids[:, :, 0].all() and solids[:, :, nz - 1].all()
