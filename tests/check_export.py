"""
Contrôle d'intégrité d'un export RoomScan (dossier du scan extrait du zip), à lancer sur le PC après une marche test.

    python tests/check_export.py <dossier du scan>

Vérifie : taille exacte de mesh.ply (en-tête + 24 NV + 14 NT), indices < NV, classes <= 7, ceilingHeight - floorHeight
> 1,8 m, cameras/*.json en 1024x768, JPEG 1024x768, profondeur uint16 192x256 (lignes x colonnes), nombre de fichiers
= keyframes de scan.json. Code de sortie 0 si tout est bon. Ce fichier n'est pas un test pytest : il prend un vrai scan.
"""
import json
import pathlib
import re
import sys

import numpy as np
from PIL import Image

FACE = np.dtype([("n", "u1"), ("idx", "<i4", (3,)), ("cls", "u1")])
VERTEX = np.dtype([("p", "<f4", (3,)), ("n", "<f4", (3,))])


def read_ply(path):
    data = path.read_bytes()
    end = data.find(b"end_header\n")
    if end < 0:
        raise ValueError("pas de end_header")
    hdr = end + len(b"end_header\n")
    text = data[:hdr].decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("format inattendu")
    nv = int(re.search(r"element vertex (\d+)", text).group(1))
    nf = int(re.search(r"element face (\d+)", text).group(1))
    return data, hdr, nv, nf


def main(scan):
    errors = []

    def check(ok, label):
        print(("ok    " if ok else "ERREUR"), label)
        if not ok:
            errors.append(label)

    meta = json.loads((scan / "scan.json").read_text(encoding="utf-8"))
    data, hdr, nv, nf = read_ply(scan / "mesh.ply")
    check(len(data) == hdr + 24 * nv + 14 * nf, f"taille mesh.ply = {len(data)} (en-tête {hdr}, NV {nv}, NT {nf})")
    if errors:
        print("\nmesh.ply tronqué : contrôles suivants abandonnés")
        return 1
    check(nv == meta["vertices"] and nf == meta["triangles"], "sommets et triangles identiques à scan.json")
    vertices = np.frombuffer(data, VERTEX, nv, hdr)
    faces = np.frombuffer(data, FACE, nf, hdr + 24 * nv)
    check(bool(np.isfinite(vertices["p"]).all()), "positions finies")
    check(bool((faces["n"] == 3).all()), "faces à 3 sommets")
    check(nf == 0 or bool((faces["idx"] >= 0).all() and (faces["idx"] < nv).all()), "indices dans [0, NV[")
    check(nf == 0 or int(faces["cls"].max()) <= 7, "classes <= 7")
    check(sum(meta["classTriangles"].values()) == nf, "classTriangles somme à NT")

    floor, ceil = meta.get("floorHeight"), meta.get("ceilingHeight")
    check(floor is not None and ceil is not None and ceil - floor > 1.8, f"hauteur sous plafond (sol {floor}, plafond {ceil})")

    kf = scan / "keyframes"
    cams = sorted((kf / "cameras").glob("*.json"))
    count = meta["keyframes"]
    for sub, ext in (("cameras", "json"), ("images", "jpg"), ("depth", "png")):
        n = len(list((kf / sub).glob(f"*.{ext}")))
        check(n == count, f"{sub}/ : {n} fichiers pour keyframes = {count}")
    check(all((c := json.loads(p.read_text()))["width"] == 1024 and c["height"] == 768 for p in cams), "cameras/*.json en 1024x768")

    bad_jpg = bad_depth = 0
    for p in cams:
        jpg, png = kf / "images" / f"{p.stem}.jpg", kf / "depth" / f"{p.stem}.png"
        if jpg.exists():
            with Image.open(jpg) as im:
                bad_jpg += im.size != (1024, 768)
        else:
            bad_jpg += 1
        if png.exists():
            d = np.asarray(Image.open(png))
            bad_depth += d.dtype != np.uint16 or d.shape != (192, 256)
        else:
            bad_depth += 1
    check(bad_jpg == 0, f"JPEG 1024x768 ({bad_jpg} incorrects)")
    check(bad_depth == 0, f"profondeur uint16 192x256 ({bad_depth} incorrectes)")

    print("\nOK" if not errors else f"\n{len(errors)} erreur(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(pathlib.Path(sys.argv[1])))
