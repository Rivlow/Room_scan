#!/usr/bin/env python3
"""
scan_to_mesh.py : maillage fin et coloré d'une pièce scannée, pour le mode d'affichage réaliste de Physical_engine.

Les cartes de profondeur LiDAR d'un export « keyframes » (RoomScan, Polycam : images/, depth/, confidence/, cameras/) sont
fusionnées dans une fonction distance signée tronquée (TSDF) sur une grille fine, restreinte à une bande autour des
surfaces observées. Chaque voxel de la bande prend la couleur moyenne des photos qui le voient, pondérée par la distance,
la position dans l'image et la netteté de la photo. La surface zéro est extraite par marching cubes.

Le maillage ne sert qu'à l'image : le calcul garde sa grille de mailles solides. Il est écrit dans le repère du domaine,
à partir de la rotation et de l'origine notées par scan_to_domain.py dans le JSON du cas.

Sortie : render_mesh.bin, à côté du JSON du cas
    char[4] "RMSH", uint32 version (1), uint32 nv, uint32 nt
    nv sommets de 32 octets : float x, y, z ; uint8 r, g, b, a (sRGB) ; float nx, ny, nz ; float 0
    nt triangles de 3 uint32 (sens direct vu de l'air)

    python scan_to_mesh.py <dossier keyframes ou zip> --case <cas.json> [--voxel 0.02]
"""
import argparse
import json
import math
import pathlib
import struct
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.sparse import csgraph


def find_keyframes(path, tmp):
    """Dossier contenant cameras/, depth/ et images/ (extrait le zip dans tmp au besoin)."""
    path = pathlib.Path(path)
    if path.is_file():
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = pathlib.Path(tmp)
    for root in [path, path / "keyframes", *path.glob("*/keyframes")]:
        if (root / "cameras").is_dir() and (root / "depth").is_dir() and (root / "images").is_dir():
            return root
    return None


def load_frame(root, cam_file):
    cam = json.loads(cam_file.read_text())
    ts = cam_file.stem
    depth_file = root / "depth" / f"{ts}.png"
    image_file = root / "images" / f"{ts}.jpg"
    if not depth_file.exists() or not image_file.exists():
        return None
    depth = np.asarray(Image.open(depth_file), dtype=np.float32) / 1000.0
    conf_file = root / "confidence" / f"{ts}.png"
    if conf_file.exists():
        conf = np.asarray(Image.open(conf_file)).astype(np.int32)
        rank = conf if conf.max() <= 2 else np.where(conf < 20, 0, np.where(conf < 200, 1, 2))
        depth = np.where(rank >= 1, depth, 0.0).astype(np.float32)
    # pixels volants : les sauts de profondeur ne sont pas des surfaces
    jump = ndimage.maximum_filter(depth, 3) - ndimage.minimum_filter(depth, 3)
    depth[jump > 0.05 + 0.03 * depth] = 0.0
    T = np.array([[cam[f"t_{r}{c}"] for c in range(4)] for r in range(3)], dtype=np.float64)
    return {
        "depth": depth,
        "rgb": np.asarray(Image.open(image_file).convert("RGB"), dtype=np.float32),
        "T": T,
        "fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"],
        "width": cam["width"], "height": cam["height"],
        "blur": float(cam.get("blur_score", 1.0)),
    }


def arkit_to_domain(yaw_deg, origin):
    """Matrice 3x3 et translation : point ARKit p -> point du domaine R p + t (repère moteur tourné, coin de la grille en 0)."""
    c, s = math.cos(-math.radians(yaw_deg)), math.sin(-math.radians(yaw_deg))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    E = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])        # ARKit -> moteur
    return Rz @ E, -np.asarray(origin, dtype=np.float64)


def backproject(f):
    """Points valides de la carte de profondeur, repère caméra ARKit (regarde vers -z)."""
    h, w = f["depth"].shape
    sx, sy = w / f["width"], h / f["height"]
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    z = f["depth"]
    ok = (z > 0.15) & (z < 5.0)
    return np.stack([(u[ok] + 0.5 - f["cx"] * sx) / (f["fx"] * sx) * z[ok],
                     -(v[ok] + 0.5 - f["cy"] * sy) / (f["fy"] * sy) * z[ok], -z[ok]], axis=-1)


def fuse(frames, R, t, shape, voxel, trunc):
    """TSDF et couleurs sur la bande autour des points observés. Rend la TSDF, les poids et la couleur en pleine grille."""
    shp = np.array(shape)
    seen = np.zeros(shape, bool)
    for f in frames:
        world = backproject(f) @ f["T"][:, :3].T + f["T"][:, 3]
        ijk = np.floor((world @ R.T + t) / voxel).astype(np.int64)
        ijk = ijk[np.all((ijk >= 0) & (ijk < shp), axis=1)]
        seen[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    r = int(math.ceil(trunc / voxel))
    band = ndimage.binary_dilation(seen, structure=np.ones((3, 3, 3), bool), iterations=r)
    idx = np.flatnonzero(band.ravel())
    print(f"bande : {len(idx)} voxels sur {seen.size}", flush=True)
    i, j, k = np.unravel_index(idx, shape)
    Q = (np.stack([i, j, k], axis=-1).astype(np.float64) + 0.5) * voxel        # centres, repère domaine

    tsdf = np.zeros(len(idx), np.float32)
    weight = np.zeros(len(idx), np.float32)
    color = np.zeros((len(idx), 3), np.float32)
    cw = np.zeros(len(idx), np.float32)
    blur_ref = float(np.median([f["blur"] for f in frames]))
    world = (Q - t) @ R                                                        # domaine -> monde ARKit
    start = time.time()
    for n, f in enumerate(frames):
        pc = (world - f["T"][:, 3]) @ f["T"][:, :3]                            # monde -> caméra
        z = -pc[:, 2]
        h, w = f["depth"].shape
        sx, sy = w / f["width"], h / f["height"]
        with np.errstate(divide="ignore", invalid="ignore"):
            uu = pc[:, 0] / z * f["fx"] * sx + f["cx"] * sx
            vv = -pc[:, 1] / z * f["fy"] * sy + f["cy"] * sy
            inside = (z > 0.15) & (uu >= 0) & (vv >= 0) & (uu < w) & (vv < h)
        vis = np.flatnonzero(inside)
        ui, vi = uu[vis].astype(np.int32), vv[vis].astype(np.int32)
        d = f["depth"][vi, ui]
        good = d > 0
        vis, d = vis[good], d[good]
        sdf = (d - z[vis]) * np.sqrt(1.0 + (pc[vis, 0] ** 2 + pc[vis, 1] ** 2) / z[vis] ** 2)
        keep = sdf > -trunc
        vis, sdf, d = vis[keep], np.minimum(sdf[keep], trunc).astype(np.float32), d[keep]
        g = (1.0 / (0.25 + d * d)).astype(np.float32)                          # le LiDAR est plus précis de près
        tsdf[vis] = (tsdf[vis] * weight[vis] + sdf * g) / (weight[vis] + g)
        weight[vis] += g
        # couleur : seulement les voxels tout près de la surface vue par cette image
        near = np.abs(sdf) < 1.5 * voxel
        cv = vis[near]
        img = f["rgb"]
        ih, iw = img.shape[:2]
        px, py = uu[cv] / sx, vv[cv] / sy
        x0 = np.clip(np.floor(px - 0.5).astype(np.int32), 0, iw - 2)
        y0 = np.clip(np.floor(py - 0.5).astype(np.int32), 0, ih - 2)
        ax = np.clip(px - 0.5 - x0, 0, 1)[:, None].astype(np.float32)
        ay = np.clip(py - 0.5 - y0, 0, 1)[:, None].astype(np.float32)
        c = (img[y0, x0] * (1 - ax) + img[y0, x0 + 1] * ax) * (1 - ay) + (img[y0 + 1, x0] * (1 - ax) + img[y0 + 1, x0 + 1] * ax) * ay
        # l'exposition automatique change d'une photo à l'autre : gain qui l'aligne sur ce qui est déjà fusionné
        known = cw[cv] > 0
        if np.count_nonzero(known) > 500:
            ratio = (color[cv[known]] / cw[cv[known]][:, None]).sum(axis=1) / np.maximum(c[known].sum(axis=1), 1.0)
            c = c * float(np.clip(np.median(ratio), 0.4, 2.5))
        # au centre de l'image et sur les photos nettes, le poids est plus fort
        rx, ry = px / iw * 2 - 1, py / ih * 2 - 1
        gc = (g[near] * np.clip(1.0 - 0.7 * (rx * rx + ry * ry), 0.05, 1.0) * min(f["blur"] / blur_ref, 2.0) ** 2).astype(np.float32)
        color[cv] += c * gc[:, None]
        cw[cv] += gc
        if n % 40 == 0:
            print(f"image {n + 1} / {len(frames)} ({time.time() - start:.0f} s)", flush=True)

    vol = np.full(shape, trunc, np.float32)
    wvol = np.zeros(shape, np.float32)
    vol.ravel()[idx] = np.where(weight > 0, tsdf, trunc)
    wvol.ravel()[idx] = weight
    # couleur prolongée vers les voxels qui n'en ont pas (la surface tombe entre deux voxels)
    has = np.zeros(shape, bool)
    has.ravel()[idx] = cw > 0
    cvol = np.zeros(shape + (3,), np.float32)
    mean = color / np.maximum(cw, 1e-6)[:, None]
    cvol[i, j, k] = mean
    nearest = ndimage.distance_transform_edt(~has, return_distances=False, return_indices=True)
    return vol, wvol, cvol[nearest[0], nearest[1], nearest[2]]


def keep_large(verts, faces, min_vertices):
    """Retire les petits morceaux flottants : rend le masque des sommets gardés et les triangles renumérotés."""
    n = len(verts)
    a = np.concatenate([faces[:, 0], faces[:, 1]])
    b = np.concatenate([faces[:, 1], faces[:, 2]])
    g = sparse.coo_matrix((np.ones(len(a), bool), (a, b)), shape=(n, n))
    count, label = csgraph.connected_components(g, directed=False)
    sizes = np.bincount(label, minlength=count)
    alive = sizes[label] >= min_vertices
    remap = np.cumsum(alive) - 1
    return alive, remap[faces[alive[faces[:, 0]]]]


def write_mesh(path, verts, colors, normals, faces):
    rec = np.zeros(len(verts), dtype=[("p", "<f4", 3), ("c", "u1", 4), ("n", "<f4", 3), ("pad", "<f4")])
    rec["p"], rec["n"] = verts, normals
    rec["c"][:, :3] = np.clip(np.round(colors), 0, 255)
    rec["c"][:, 3] = 255
    with open(path, "wb") as f:
        f.write(b"RMSH" + struct.pack("<III", 1, len(verts), len(faces)))
        f.write(rec.tobytes())
        f.write(faces.astype("<u4").tobytes())


def build_mesh(source, case_path, voxel=0.02, trunc_voxels=3.0):
    """Écrit render_mesh.bin à côté du cas ; False si le scan n'a pas le format voulu ou si scikit-image manque."""
    try:
        from skimage import measure
    except ImportError:
        print("scikit-image absent : pas de maillage fin, l'affichage réaliste lissera les mailles solides")
        return False
    case_path = pathlib.Path(case_path)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    scan = case["scan"]
    lo = scan["origin_arkit_floor"]
    origin = np.array([lo[0], lo[1], lo[2] - float(scan["dx"])])
    with tempfile.TemporaryDirectory() as tmp:
        root = find_keyframes(source, tmp)
        if root is None:
            print("format de scan sans cameras/, depth/, images/ : pas de maillage fin")
            return False
        frames = [f for f in (load_frame(root, p) for p in sorted((root / "cameras").glob("*.json"))) if f is not None]
    print(f"{len(frames)} images", flush=True)
    R, t = arkit_to_domain(scan["yaw_deg"], origin)
    g = case["grid"]
    shape = tuple(int(math.ceil(g[L] / voxel)) for L in ("Lx", "Ly", "Lz"))
    print(f"grille fine {shape}, voxel {voxel * 100:.1f} cm", flush=True)
    vol, wvol, cvol = fuse(frames, R, t, shape, voxel, trunc_voxels * voxel)
    vol = ndimage.gaussian_filter(vol, 1.0)                                    # ôte le grain du LiDAR
    mask = ndimage.binary_dilation(wvol > 0.5, structure=np.ones((3, 3, 3), bool))
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, mask=mask)
    print(f"marching cubes : {len(verts)} sommets, {len(faces)} triangles", flush=True)
    # bords de scan : les surfaces vues une seule fois, de biais, font des peignes de triangles
    seen = ndimage.map_coordinates(wvol, verts.T, order=1)
    edge = np.maximum.reduce([np.linalg.norm(verts[faces[:, a]] - verts[faces[:, (a + 1) % 3]], axis=1) for a in range(3)])
    faces = faces[(seen[faces].min(axis=1) > 0.4) & (edge < 2.5)]
    alive, faces = keep_large(verts, faces, 800)
    verts = verts[alive]
    grad = np.gradient(ndimage.gaussian_filter(vol, 1.5))                      # normales plus douces que la surface
    coords = verts.T
    n = np.stack([ndimage.map_coordinates(grad[a], coords, order=1) for a in range(3)], axis=-1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)            # TSDF positive dans l'air : vers l'air
    cols = np.stack([ndimage.map_coordinates(cvol[..., a], coords, order=1) for a in range(3)], axis=-1)
    if cols.max() < 1.0:
        raise RuntimeError("couleurs du maillage toutes nulles")
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), n[faces[:, 0]]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    out = case_path.parent / "render_mesh.bin"
    write_mesh(out, (verts + 0.5) * voxel, cols, n, faces)
    print(f"{out} : {len(verts)} sommets, {len(faces)} triangles, {out.stat().st_size / 1e6:.0f} Mo", flush=True)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keyframes", help="dossier keyframes ou zip de l'export")
    ap.add_argument("--case", required=True, help="JSON du cas écrit par scan_to_domain.py")
    ap.add_argument("--voxel", type=float, default=0.02, help="taille du voxel de fusion (m), défaut 0.02")
    a = ap.parse_args()
    sys.exit(0 if build_mesh(a.keyframes, a.case, a.voxel) else 1)
