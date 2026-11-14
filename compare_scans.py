"""
compare_scans.py : compare deux versions d'un domaine (jumeau numérique) : ce qui est apparu, ce qui a disparu.

Les deux domaines (dossiers écrits par scan_to_domain.py : <nom>.json + solids.npy) doivent partager le même repère
ARKit : scan repris dans RoomScan (« Reprendre le scan », carte ARKit), ou second scan recalé sur le premier par
register_scans.py et converti avec lui en référence (python scan_to_domain.py <ref> <scan2> …, ou le scan2 seul
après avoir copié son alignment.json : voir README). Chaque domaine a sa propre grille (orientation des murs, emprise,
taille de maille) : le nouveau est rééchantillonné sur la grille de la référence.

    python compare_scans.py <ref>/<ref>.json <nouveau>/<nouveau>.json [--out <dossier>] [--tol 1]

Sorties (--out, défaut : dossier du nouveau domaine) :
    diff.npy            uint8 sur la grille de la référence : 0 air dans les deux, 1 solide dans les deux,
                        2 apparu (solide seulement dans le nouveau), 3 disparu (solide seulement dans la référence),
                        4 hors de la grille du nouveau (non comparé) ; dalles sol / plafond : 1
    diff.json           volumes (m3), part des mailles comparées, liste des zones changées (boîte englobante en m
                        dans le repère du domaine de référence : origine au coin de sa grille, sol à z = dx, comme
                        les aperçus et le cas JSON ; volume), paramètres
    apercu_diff.png     vue de dessus : gris solide inchangé, vert apparu, rouge disparu, bleu clair non comparé
    rapport_diff.txt    le résumé lisible
--tol : tolérance en mailles (défaut 1) : une maille solide n'est « apparue » que si aucune maille solide de la
référence n'est à moins de tol mailles (et réciproquement) ; absorbe le bruit de recalage et de voxelisation.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def load_domain(case_path):
    case = json.loads(case_path.read_text(encoding="utf-8"))
    solid = np.load(case_path.parent / case["solids"]["file"]) > 0
    sc = case["scan"]
    dx = float(sc["dx"])
    yaw = math.radians(sc["yaw_deg"])
    c, s = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])                     # moteur -> repère tourné (points @ Rz.T)
    o = sc["origin_arkit_floor"]
    lo = np.array([o[0], o[1], o[2] - dx])                                 # coin de la maille (0, 0, 0), repère tourné
    return dict(case=case, path=case_path, solid=solid, dx=dx, Rz=Rz, lo=lo, shape=np.array(solid.shape))


def resample(new, ref):
    """Solides du nouveau domaine sur la grille de la référence (plus proche maille), et masque des mailles de la
    référence couvertes par la grille du nouveau. Passage par le repère du moteur commun aux deux scans."""
    nx, ny, nz = ref["shape"]
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    centers = np.stack([i, j, k], -1).reshape(-1, 3) + 0.5
    q = ref["lo"] + centers * ref["dx"]                                    # repère tourné de la référence
    p = q @ ref["Rz"]                                                      # = Rz^T q : repère du moteur
    qn = p @ new["Rz"].T                                                   # repère tourné du nouveau
    ijk = np.floor((qn - new["lo"]) / new["dx"]).astype(int)
    inside = (ijk >= 0).all(1) & (ijk < new["shape"]).all(1)
    out = np.zeros(len(centers), bool)
    sel = ijk[inside]
    out[inside] = new["solid"][sel[:, 0], sel[:, 1], sel[:, 2]]
    return out.reshape(nx, ny, nz), inside.reshape(nx, ny, nz)


def compare(ref, new, tol):
    """Code par maille de la référence (voir l'en-tête) ; une maille solide d'un seul côté mais à moins de tol mailles
    d'une maille solide de l'autre côté compte comme inchangée."""
    a = ref["solid"]
    b, covered = resample(new, ref)
    nz = a.shape[2]
    inner = np.zeros_like(a)
    inner[:, :, 1:nz - 1] = True                                           # hors dalles sol / plafond
    cube = np.ones((3, 3, 3), bool)
    a_near = ndimage.binary_dilation(a, structure=cube, iterations=tol) if tol > 0 else a
    b_near = ndimage.binary_dilation(b, structure=cube, iterations=tol) if tol > 0 else b
    diff = np.full(a.shape, 4, np.uint8)
    diff[covered] = np.where((a | b)[covered], 1, 0)
    diff[covered & inner & b & ~a_near] = 2
    diff[covered & inner & a & ~b_near] = 3
    diff[~inner] = 1
    return diff


def zones(diff, code, dx, min_cells):
    """Composantes connexes des mailles `code` : boîte englobante en m dans le repère du domaine de référence (celui
    du cas JSON et des aperçus : origine au coin de la grille, murs alignés sur les axes, sol à z = dx), volume (m3)."""
    lab, n = ndimage.label(diff == code)
    out = []
    if n == 0:
        return out
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    objects = ndimage.find_objects(lab)
    for idx in np.argsort(-sizes):
        if sizes[idx] < min_cells:
            break
        sl = objects[idx]
        out.append({"cells": int(sizes[idx]), "volume_m3": round(float(sizes[idx]) * dx ** 3, 4),
                    "box_min": [round(s.start * dx, 3) for s in sl],
                    "box_max": [round(s.stop * dx, 3) for s in sl]})
    return out


def save_preview(path, diff):
    nx, ny, nz = diff.shape
    img = np.full((nx, ny, 3), 255, np.uint8)
    inner = diff[:, :, 1:nz - 1]
    unchanged = (inner == 1).any(2)
    img[unchanged] = (170, 170, 170)
    img[(inner == 4).all(2)] = (210, 230, 250)
    img[(inner == 3).any(2)] = (220, 50, 50)
    img[(inner == 2).any(2)] = (40, 170, 60)
    scale = max(1, int(round(600 / max(nx, ny))))
    Image.fromarray(np.flipud(img.transpose(1, 0, 2))).resize((nx * scale, ny * scale), Image.NEAREST).save(path)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Différences entre deux versions d'un domaine scanné")
    ap.add_argument("reference", type=pathlib.Path, help="<ref>.json du domaine de référence (scan_to_domain.py)")
    ap.add_argument("new", type=pathlib.Path, help="<nouveau>.json de la nouvelle version, même repère ARKit")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="dossier des sorties (défaut : celui du nouveau)")
    ap.add_argument("--tol", type=int, default=1, help="tolérance en mailles de la référence (défaut 1)")
    ap.add_argument("--min-zone", type=float, default=0.01, help="volume minimal (m3) d'une zone listée, défaut 0.01")
    a = ap.parse_args()
    out = a.out or a.new.parent
    out.mkdir(parents=True, exist_ok=True)
    ref, new = load_domain(a.reference), load_domain(a.new)
    diff = compare(ref, new, a.tol)
    np.save(out / "diff.npy", diff)
    save_preview(out / "apercu_diff.png", diff)
    dx = ref["dx"]
    nz = diff.shape[2]
    inner = diff[:, :, 1:nz - 1]
    counts = {int(c): int((inner == c).sum()) for c in range(5)}
    compared = inner.size - counts[4]
    min_cells = max(1, int(round(a.min_zone / dx ** 3)))
    added = zones(diff, 2, dx, min_cells)
    removed = zones(diff, 3, dx, min_cells)
    info = {
        "reference": str(a.reference), "new": str(a.new), "dx": dx, "tol_cells": a.tol,
        "grid": [int(v) for v in diff.shape], "compared_share": round(compared / inner.size, 4),
        "added_m3": round(counts[2] * dx ** 3, 4), "removed_m3": round(counts[3] * dx ** 3, 4),
        "solid_both_m3": round(counts[1] * dx ** 3, 4),
        "added_zones": added, "removed_zones": removed,
    }
    (out / "diff.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"référence              {a.reference}",
        f"nouveau                {a.new}",
        f"grille de référence    {diff.shape[0]} x {diff.shape[1]} x {diff.shape[2]}, dx = {dx} m, tolérance {a.tol} maille(s)",
        f"mailles comparées      {100 * compared / inner.size:.1f} % (le reste hors de la grille du nouveau)",
        f"solide inchangé        {counts[1] * dx ** 3:.3f} m3",
        f"apparu                 {counts[2] * dx ** 3:.3f} m3 en {len(added)} zone(s) >= {a.min_zone} m3",
        f"disparu                {counts[3] * dx ** 3:.3f} m3 en {len(removed)} zone(s) >= {a.min_zone} m3",
    ]
    for name, zs in (("apparu", added), ("disparu", removed)):
        for z in zs[:10]:
            lines.append(f"  {name:8s} {z['volume_m3']:.3f} m3  x {z['box_min'][0]:.2f}..{z['box_max'][0]:.2f}  "
                         f"y {z['box_min'][1]:.2f}..{z['box_max'][1]:.2f}  z {z['box_min'][2]:.2f}..{z['box_max'][2]:.2f}")
    lines.append(f"sorties                {out / 'diff.npy'}, {out / 'diff.json'}, {out / 'apercu_diff.png'}")
    (out / "rapport_diff.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
