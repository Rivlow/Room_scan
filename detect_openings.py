"""
detect_openings.py : portes et fenêtres d'un scan RoomScan  ->  rectangles de mailles d'un domaine de Physical_engine.

Le maillage ARKit (mesh.ply) porte une classe par triangle, dont « window » (6) et « door » (7). Ces triangles sont
projetés sur la grille du domaine (même rotation et même origine que scan_to_domain.py, lues dans "scan" du JSON),
regroupés en objets, ramenés au plan de leur mur, filtrés, puis régularisés en un rectangle d'une maille d'épaisseur
posé sur la couche du mur qui touche l'air de la pièce. Chaque objet est écrit dans le JSON du domaine :

    "openings": [ { "kind": "door" | "window", "name": "Porte 1", "face": 1..6, "box": [i0, j0, k0, i1, j1, k1],
                    "width": 0.9, "height": 2.0, "sill": 0.0 } ]

face : face active tournée vers la pièce (1 -x, 2 +x, 3 -y, 4 +y) ; box : indices inclus. Physical_engine propose ces
objets dans « Bords et ouvertures » : choisir une condition crée la zone peinte correspondante.

Filtres (la classification d'ARKit déborde et se trompe parfois) : au moins 0,25 m² de triangles dans le plan du mur,
0,6 m de haut et 0,4 m de large ; la partie hors du plan (un battant ouvert) est ignorée.

Usage :
    python detect_openings.py <domaine>.json [--mesh mesh.ply]      ajoute "openings" au JSON, apercu_ouvertures.png
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CLASSES = {7: ("door", "Porte"), 6: ("window", "Fenêtre")}
MIN_AREA = 0.25          # m² de triangles dans le plan du mur
MIN_HEIGHT = 0.6         # m
MIN_WIDTH = 0.4          # m
PLANE_TOL = 2            # mailles de part et d'autre du plan du mur
MERGE = 6                # mailles : morceaux d'un même mur fusionnés à cette distance
FLOOR_SNAP = 0.15        # m : un bas d'objet plus près du sol est posé sur le sol


def read_ply(path):
    """Maillage ARKit de RoomScan -> sommets (nv, 6) : position puis normale (repère ARKit), faces (indices i, classe c)."""
    raw = pathlib.Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii", "replace")
    if "binary_little_endian" not in head or "property uchar class" not in head:
        raise ValueError(f"{path} : PLY binaire RoomScan attendu (avec une classe par face)")
    nv = int(head.split("element vertex ")[1].split()[0])
    nf = int(head.split("element face ")[1].split()[0])
    V = np.frombuffer(raw, np.float32, nv * 6, end).reshape(nv, 6)
    F = np.frombuffer(raw, np.dtype([("n", "u1"), ("i", "<i4", 3), ("c", "u1")]), nf, end + nv * 24)
    if not (F["n"] == 3).all():
        raise ValueError(f"{path} : faces non triangulaires")
    return V.astype(np.float64), F


def write_ply_mesh(path, V, F):
    """Écrit un maillage au format de RoomScan (voir read_ply) : V (nv, 6) positions et normales, F faces (i, c)."""
    nv, nf = len(V), len(F)
    head = (f"ply\nformat binary_little_endian 1.0\ncomment RoomScan ARKit mesh, ARKit frame (x right, y up, z backward), metres\n"
            f"comment class: 0 none 1 wall 2 floor 3 ceiling 4 table 5 seat 6 window 7 door\nelement vertex {nv}\n"
            + "".join(f"property float {p}\n" for p in ("x", "y", "z", "nx", "ny", "nz"))
            + f"element face {nf}\nproperty list uchar int vertex_indices\nproperty uchar class\nend_header\n")
    out = np.zeros(nf, np.dtype([("n", "u1"), ("i", "<i4", 3), ("c", "u1")]))
    out["n"], out["i"], out["c"] = 3, F["i"], F["c"]
    pathlib.Path(path).write_bytes(head.encode("ascii") + np.ascontiguousarray(V, "<f4").tobytes() + out.tobytes())


def find_mesh(*dirs):
    """mesh.ply du scan RoomScan : dans le dossier keyframes ou son parent."""
    for d in dirs:
        if d is None:
            continue
        d = pathlib.Path(d)
        for p in (d / "mesh.ply", d.parent / "mesh.ply"):
            if p.exists():
                return p
    return None


def grid_frame(case):
    sc, g = case["scan"], case["grid"]
    dx = float(sc["dx"])
    yaw = math.radians(sc["yaw_deg"])
    c, s = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o = sc["origin_arkit_floor"]
    lo = np.array([o[0], o[1], o[2] - dx])
    return dx, Rz, lo, (int(g["nx"]), int(g["ny"]), int(g["nz"]))


def splat(P, F, tri, lo, dx, shape):
    """Aire des triangles par maille (échantillonnage barycentrique à dx / 3)."""
    A, B, C = (P[F["i"][tri, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    L = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - A, axis=1), np.linalg.norm(C - B, axis=1)])
    m = np.maximum(1, np.ceil(L / (dx / 3)).astype(int))
    acc = np.zeros(int(np.prod(shape)))
    for mm in np.unique(m):
        sel = m == mm
        u, v = np.meshgrid(np.arange(mm + 1), np.arange(mm + 1), indexing="ij")
        ok = u + v <= mm
        u, v = u[ok] / mm, v[ok] / mm
        pts = A[sel, None] + u[None, :, None] * (B[sel, None] - A[sel, None]) + v[None, :, None] * (C[sel, None] - A[sel, None])
        w = np.repeat((area[sel] / len(u))[:, None], len(u), 1)
        ijk = np.floor((pts.reshape(-1, 3) - lo) / dx).astype(int)
        ins = np.all((ijk >= 0) & (ijk < shape), 1)
        acc += np.bincount(np.ravel_multi_index(ijk[ins].T, shape), w.ravel()[ins], acc.size)
    return acc.reshape(shape)


def plane_of(mask, acc, dx):
    """Plan vertical aligné sur les axes (scan_to_domain aligne les murs) : axe par ACP, position = médiane pondérée."""
    ijk = np.argwhere(mask)
    w = acc[mask]
    x = (ijk + 0.5) * dx
    mu = (x * w[:, None]).sum(0) / w.sum()
    cov = ((x - mu).T * w) @ (x - mu) / w.sum()
    n = np.linalg.eigh(cov)[1][:, 0]
    ax = int(np.argmax(np.abs(n[:2])))
    o = np.argsort(ijk[:, ax])
    cw = np.cumsum(w[o])
    return ax, int(ijk[o][np.searchsorted(cw, 0.5 * cw[-1]), ax])


def wpercentile(v, w, q):
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return v[o][min(np.searchsorted(cw, q / 100.0 * cw[-1]), len(v) - 1)]


def detect(mesh_path, case, solid, log=print):
    """Liste des ouvertures (dictionnaires du format "openings") du domaine décrit par `case` et son masque `solid`."""
    dx, Rz, lo, shape = grid_frame(case)
    if solid.shape != shape:
        raise ValueError(f"masque {solid.shape} différent de la grille {shape}")
    V, F = read_ply(mesh_path)
    P = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1) @ Rz.T                 # ARKit -> moteur, murs sur les axes
    lab_air, _ = ndimage.label(~solid)
    room = lab_air == np.bincount(lab_air[lab_air > 0]).argmax()        # l'air de la pièce
    coords = np.indices(shape)
    found = []
    for cls, (kind, _) in CLASSES.items():
        tri = np.flatnonzero(F["c"] == cls)
        if len(tri) == 0:
            continue
        acc = splat(P, F, tri, lo, dx, shape)
        hit = acc > 0
        lab, nl = ndimage.label(ndimage.binary_dilation(hit, np.ones((3, 3, 3), bool)), np.ones((3, 3, 3), bool))
        lab = np.where(hit, lab, 0)
        # morceaux ramenés au plan de leur mur
        pieces = []
        for l in range(1, nl + 1):
            m = lab == l
            ax, pos = plane_of(m, acc, dx)
            pieces.append(dict(ax=ax, pos=pos, mask=m & (np.abs(coords[ax] - pos) <= PLANE_TOL)))
        # morceaux d'un même mur et proches : un seul objet
        parent = list(range(len(pieces)))
        find = lambda i: i if parent[i] == i else find(parent[i])
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                p, q = pieces[i], pieces[j]
                if p["ax"] != q["ax"] or abs(p["pos"] - q["pos"]) > PLANE_TOL or not p["mask"].any() or not q["mask"].any():
                    continue
                if (ndimage.binary_dilation(p["mask"], np.ones((3, 3, 3), bool), iterations=MERGE) & q["mask"]).any():
                    parent[find(j)] = find(i)
        groups = {}
        for i, p in enumerate(pieces):
            groups.setdefault(find(i), []).append(p)
        for members in groups.values():
            m = np.logical_or.reduce([p["mask"] for p in members])
            if not m.any():
                continue
            ax = members[0]["ax"]
            other = 1 - ax
            ijk = np.argwhere(m)
            w = acc[m]
            area = float(w.sum())
            # rectangle : percentiles pondérés dans le plan (le contour d'ARKit est irrégulier)
            u0, u1 = int(wpercentile(ijk[:, other], w, 1)), int(wpercentile(ijk[:, other], w, 99))
            k0, k1 = int(wpercentile(ijk[:, 2], w, 2)), int(wpercentile(ijk[:, 2], w, 98))
            k0 = max(k0, 1)
            k1 = min(k1, shape[2] - 2)
            if (k0 - 1) * dx < FLOOR_SNAP:
                k0 = 1                                                   # posé sur le sol (k = 0 : dalle)
            width, height, sill = (u1 - u0 + 1) * dx, (k1 - k0 + 1) * dx, (k0 - 1) * dx
            why = [t for t, bad in ((f"aire {area:.2f} m²", area < MIN_AREA), (f"hauteur {height:.2f} m", height < MIN_HEIGHT),
                                    (f"largeur {width:.2f} m", width < MIN_WIDTH)) if bad]
            # côté pièce : de l'air de la pièce de part et d'autre du plan ; des deux côtés (passage) : vers le centre
            pos = int(np.median(ijk[:, ax]))
            sl = [slice(None)] * 3
            sl[other] = slice(u0, u1 + 1)
            sl[2] = slice(k0, k1 + 1)
            def air_frac(layer):
                if not 0 <= layer < shape[ax]:
                    return 0.0
                s = list(sl)
                s[ax] = layer
                return float(room[tuple(s)].mean())
            cnt = [sum(air_frac(pos + sg * d) for d in range(2, 7)) for sg in (-1, 1)]
            if min(cnt) > 0.6 * max(cnt):
                plus = np.argwhere(room)[:, ax].mean() > pos
            else:
                plus = cnt[1] >= cnt[0]
            dirn = 1 if plus else -1
            # couche peinte : à 2 mailles du plan au plus, sa voisine côté pièce dans l'air de la pièce
            cands = [p for p in range(pos - PLANE_TOL, pos + PLANE_TOL + 1) if 0 <= p < shape[ax] and 0 <= p + dirn < shape[ax]]
            good = [p for p in cands if air_frac(p + dirn) >= 0.5]
            layer = min(good, key=lambda p: (abs(p - pos), dirn * p)) if good else max(cands, key=lambda p: air_frac(p + dirn))
            if air_frac(layer + dirn) < 0.3:
                why.append("pas d'air de la pièce devant")
            box = [0, 0, 0, 0, 0, 0]
            box[ax], box[3 + ax] = layer, layer
            box[other], box[3 + other] = u0, u1
            box[2], box[5] = k0, k1
            d = dict(kind=kind, face=2 * ax + (2 if plus else 1), box=[int(b) for b in box], width=round(width, 3),
                     height=round(height, 3), sill=round(sill, 3), area=area, why=why)
            found.append(d)
            if why:
                log(f"  écarté : {kind} {width:.2f} x {height:.2f} m ({', '.join(why)})")
    kept = [d for d in found if not d["why"]]
    out = []
    for cls, (kind, label) in CLASSES.items():
        same = sorted((d for d in kept if d["kind"] == kind), key=lambda d: -d["area"])
        for n, d in enumerate(same, 1):
            out.append(dict(kind=kind, name=f"{label} {n}", face=d["face"], box=d["box"], width=d["width"], height=d["height"], sill=d["sill"]))
    walls = {1: "est", 2: "ouest", 3: "nord", 4: "sud"}        # la pièce est du côté de la face active : le mur est à l'opposé
    for d in out:
        log(f"  {d['name']} : {d['width']:.2f} x {d['height']:.2f} m, mur {walls[d['face']]}, à {d['sill']:.2f} m du sol")
    return out


def save_preview(path, openings, solid, colors=None, scale=6):
    """Élévation de chaque ouverture vue depuis la pièce, rectangle retenu sur les couleurs du scan."""
    tiles = []
    for d in openings:
        b = d["box"]
        ax = (d["face"] - 1) // 2
        other = 1 - ax
        toward = 1 if d["face"] % 2 == 0 else -1
        layer = b[ax]
        n = solid.shape[ax]
        # premier solide vu depuis la pièce, de 3 mailles devant la couche peinte vers l'extérieur
        start = min(max(layer + 3 * toward, 0), n - 1)
        rng = range(start, -1, -1) if toward > 0 else range(start, n)
        img = np.full((solid.shape[other], solid.shape[2], 3), 235, np.uint8)
        filled = np.zeros(img.shape[:2], bool)
        for t in rng:
            sol = solid.take(t, axis=ax) & ~filled
            if colors is not None:
                img[sol] = colors.take(t, axis=ax)[sol][:, :3]
            else:
                img[sol] = 120
            filled |= sol
        u0, u1, k0, k1 = b[other], b[3 + other], b[2], b[5]
        img[u0:u1 + 1, [k0, k1]] = (230, 60, 40)
        img[[u0, u1], k0:k1 + 1] = (230, 60, 40)
        a0, a1 = max(u0 - 20, 0), min(u1 + 21, solid.shape[other])
        t = img[a0:a1]
        if toward < 0:
            t = t[::-1]                    # vu depuis la pièce
        tile = Image.fromarray(np.flipud(t.transpose(1, 0, 2))).resize(((a1 - a0) * scale, solid.shape[2] * scale), Image.NEAREST)
        ImageDraw.Draw(tile).text((4, 4), d["name"], fill=(255, 255, 0))
        tiles.append(tile)
    if not tiles:
        return False
    W = sum(t.width for t in tiles) + 8 * (len(tiles) - 1)
    sheet = Image.new("RGB", (W, max(t.height for t in tiles)), (255, 255, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + 8
    sheet.save(path)
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Portes et fenêtres d'un scan RoomScan -> \"openings\" du domaine")
    ap.add_argument("case", type=pathlib.Path, help="JSON du domaine écrit par scan_to_domain.py")
    ap.add_argument("--mesh", type=pathlib.Path, default=None,
                    help="mesh.ply du scan (défaut : \"scan.mesh\" à côté du cas, sinon à côté de \"scan.source\")")
    a = ap.parse_args()
    case = json.loads(a.case.read_text(encoding="utf-8"))
    merged = a.case.parent / case["scan"]["mesh"] if case.get("scan", {}).get("mesh") else None
    mesh = a.mesh or (merged if merged is not None and merged.exists() else None) or find_mesh(case.get("scan", {}).get("source"))
    if mesh is None:
        sys.exit("mesh.ply introuvable : indiquer --mesh")
    folder = a.case.parent
    solid = np.load(folder / case["solids"]["file"]) > 0
    colors = np.load(folder / case["solids"]["colors"]) if "colors" in case["solids"] else None
    print(f"ouvertures de {mesh}")
    openings = detect(mesh, case, solid)
    case["openings"] = openings
    a.case.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    if save_preview(folder / "apercu_ouvertures.png", openings, solid, colors):
        print(f"aperçu : {folder / 'apercu_ouvertures.png'}")
    print(f"{len(openings)} ouverture(s) écrite(s) dans {a.case}")


if __name__ == "__main__":
    main()
