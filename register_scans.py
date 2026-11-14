"""
register_scans.py : recalage global de plusieurs scans dans le repère du premier.

Deux scans du même lieu faits à des moments différents (sans reprise par la carte ARKit) ont chacun leur repère
ARKit : même échelle métrique, même verticale (gravité), mais une origine et une orientation horizontale quelconques.
Ce script retrouve la transformation (rotation autour de la verticale + translation) qui amène chaque scan dans le
repère du scan de référence, et l'écrit dans alignment.json à côté du scan. scan_to_domain.py, appelé ensuite avec
plusieurs scans, l'applique pour sculpter un seul domaine :

    python register_scans.py <scan_ref> <scan2> [<scan3> ...] [--voxel 0.05] [--stride 4]
    python scan_to_domain.py <scan_ref> <scan2> ... --out <dossier> --name appartement

Méthode :
  1. nuage de points LiDAR de chaque scan (comme scan_to_domain : pixels fiables projetés par les poses ARKit), réduit
     à une maille de --voxel ;
  2. a priori : la verticale est connue, le sol de chaque scan donne le décalage vertical, l'orientation des murs
     (histogramme des normales) donne la rotation à un quart de tour près : quatre candidats ;
  3. pour chaque candidat, la translation horizontale vient de la corrélation croisée (FFT) des vues de dessus des
     surfaces verticales (murs, meubles) ;
  4. chaque candidat est affiné par ICP point-plan (Open3D), le meilleur recouvrement gagne. Sans Open3D l'étape
     grossière seule est écrite (précision : une maille de --cell), avec un avertissement.
  --global : recalage sans a priori (descripteurs FPFH + RANSAC, Open3D), pour des scans où les murs ne dominent pas ;
  la rotation trouvée est ensuite projetée sur la verticale, puis ICP.

Sorties, à côté de chaque scan recalé (dossier du scan, ou dossier keyframes pour Polycam / SpatialCapture) :
    alignment.json      "transform_arkit" : matrice 4x4 (lignes) qui envoie les coordonnées ARKit du scan dans le
                        repère ARKit de la référence ; "transform_engine" : la même dans le repère du moteur (z haut) ;
                        "fitness" : part des points du scan à moins d'une maille d'un point de la référence après
                        recalage (le recouvrement), "rmse" (m), "yaw_deg", "translation", "reference"
    alignment.png       vue de dessus : référence en gris, scan recalé en rouge, recouvrement en noir
"""
import argparse
import atexit
import datetime
import json
import math
import pathlib
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import signal

import scan_to_domain as sd

E_ENGINE = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # p_moteur = E p_arkit (scan_to_domain.to_engine)


# ---------------------------------------------------------------------------------------------------------------
# nuages

def voxel_down(points, normals, voxel):
    """Un point par maille de `voxel` m : centre de gravité des points, normale moyenne (unitaire)."""
    ijk = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(ijk, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n).astype(float)
    P = np.stack([np.bincount(inv, weights=points[:, k], minlength=n) for k in range(3)], 1) / cnt[:, None]
    N = np.stack([np.bincount(inv, weights=normals[:, k], minlength=n) for k in range(3)], 1)
    L = np.linalg.norm(N, axis=1, keepdims=True)
    ok = L[:, 0] > 1e-9
    N[ok] /= L[ok]
    return P[ok], N[ok]


def load_cloud(path, a, work):
    """Nuage d'un scan dans le repère du moteur (z haut, repère ARKit du scan) : points et normales réduits, origines
    des caméras, altitude du sol et orientation des murs (a priori du recalage)."""
    root = sd.resolve_source(path, work)
    frames = sd.load_frames(root, a.min_conf, a.max_depth, a.stride, a.move, a.turn, color=False)
    pts = np.concatenate([f["points"] for f in frames])
    nrm = np.concatenate([f["normals"] for f in frames])
    origins = np.array([f["origin"] for f in frames])
    del frames
    floors = sd.horizontal_planes(pts[:, 2], nrm, +1)
    if not floors:
        sys.exit(f"{path} : sol introuvable, le recalage s'appuie sur le sol")
    floor = min(z for z, _ in floors)
    yaw = sd.dominant_yaw(nrm)
    P, N = voxel_down(pts, nrm, a.voxel)
    print(f"{path.name} : {len(pts)} points LiDAR -> {len(P)} à {a.voxel} m, sol {floor:.3f} m, murs {math.degrees(yaw):.1f} deg")
    return dict(path=path, root=root, points=P, normals=N, origins=origins, floor=floor, yaw=yaw)


# ---------------------------------------------------------------------------------------------------------------
# transformations (repère du moteur, 4x4)

def rot_z(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def to_arkit(T_engine):
    """Même transformation exprimée sur des coordonnées ARKit : M_ark = E^T M_eng E."""
    M = np.eye(4)
    M[:3, :3] = E_ENGINE.T @ T_engine[:3, :3] @ E_ENGINE
    M[:3, 3] = E_ENGINE.T @ T_engine[:3, 3]
    return M


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------------------------------------------------------
# étape grossière : a priori sol + murs, corrélation des vues de dessus

def top_view(points, normals, floor, cell, lo=None, shape=None):
    """Image binaire de dessus des surfaces verticales (|nz| < 0.6) entre 0,1 et 2,5 m au-dessus du sol."""
    sel = (np.abs(normals[:, 2]) < 0.6) & (points[:, 2] > floor + 0.1) & (points[:, 2] < floor + 2.5)
    xy = points[sel, :2]
    if lo is None:
        lo = np.floor(xy.min(0) / cell) * cell - cell
        shape = tuple((np.ceil((xy.max(0) - lo) / cell) + 2).astype(int))
    ij = np.floor((xy - lo) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    img = np.zeros(shape, np.float32)
    img[ij[ok, 0], ij[ok, 1]] = 1
    return img, lo


def coarse_candidates(ref, scan, cell):
    """Quatre transformations candidates (rotation = écart des orientations de murs + k × 90°) ; pour chacune, le
    décalage vertical aligne les sols et la translation horizontale maximise la corrélation des vues de dessus.
    Score : part des cellules occupées qui coïncident (sur le plus petit des deux scans)."""
    img_ref, lo_ref = top_view(ref["points"], ref["normals"], ref["floor"], cell)
    out = []
    for k in range(4):
        yaw = ref["yaw"] - scan["yaw"] + k * math.pi / 2
        R = rot_z(yaw)
        P = scan["points"] @ R.T
        N = scan["normals"] @ R.T
        P[:, 2] += ref["floor"] - scan["floor"]
        img, lo = top_view(P, N, ref["floor"], cell)
        corr = signal.correlate(img_ref, img, mode="full", method="fft")
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        shift = np.array([i - (img.shape[0] - 1), j - (img.shape[1] - 1)])
        t_xy = lo_ref + shift * cell - lo
        score = float(corr[i, j]) / max(min(img_ref.sum(), img.sum()), 1.0)
        t = np.array([t_xy[0], t_xy[1], ref["floor"] - scan["floor"]])
        out.append(dict(T=make_T(R, t), yaw=yaw, score=score))
    return out


# ---------------------------------------------------------------------------------------------------------------
# affinage : Open3D

def open3d_or_none():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        return None


def to_pcd(o3d, P, N):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    return pcd


def icp(o3d, src, dst, T0, voxel):
    """ICP point-plan, grossier (3 mailles) puis fin (1 maille) ; (T, fitness, rmse) au pas fin."""
    T = T0
    reg = None
    for dist in (3 * voxel, voxel):
        reg = o3d.pipelines.registration.registration_icp(
            src, dst, dist, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = np.array(reg.transformation)
    return T, float(reg.fitness), float(reg.inlier_rmse)


def global_registration(o3d, src, dst, voxel):
    """Recalage sans a priori : descripteurs FPFH, RANSAC sur les correspondances."""
    reg_mod = o3d.pipelines.registration
    feats = []
    for pcd in (src, dst):
        feats.append(reg_mod.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100)))
    dist = 3 * voxel
    reg = reg_mod.registration_ransac_based_on_feature_matching(
        src, dst, feats[0], feats[1], True, dist, reg_mod.TransformationEstimationPointToPoint(False), 3,
        [reg_mod.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg_mod.CorrespondenceCheckerBasedOnDistance(dist)],
        reg_mod.RANSACConvergenceCriteria(200000, 0.999))
    return np.array(reg.transformation), float(reg.fitness)


def project_to_vertical(T):
    """Garde la rotation autour de z (la verticale est connue par ARKit) et la translation."""
    return make_T(rot_z(yaw_of(T[:3, :3])), T[:3, 3])


def fitness_numpy(ref, P, voxel):
    """Sans Open3D : part des points de P à moins d'une maille d'un point de ref (sur la grille de voxel)."""
    keys = {tuple(k) for k in np.floor(ref["points"] / voxel).astype(np.int64)}
    q = np.floor(P / voxel).astype(np.int64)
    return float(np.mean([tuple(k) in keys for k in q]))


# ---------------------------------------------------------------------------------------------------------------
# sortie

def write_preview(path, ref, P, N, cell):
    """alignment.png : référence en gris, scan recalé en rouge, coïncidence en noir (vue de dessus, x à droite, y en haut)."""
    allxy = np.concatenate([ref["points"][:, :2], P[:, :2]])
    lo = np.floor(allxy.min(0) / cell) * cell - cell
    shape = tuple((np.ceil((allxy.max(0) - lo) / cell) + 2).astype(int))
    a, _ = top_view(ref["points"], ref["normals"], ref["floor"], cell, lo, shape)
    b, _ = top_view(P, N, ref["floor"], cell, lo, shape)
    rgb = np.full(shape + (3,), 255, np.uint8)
    rgb[a > 0] = (150, 150, 150)
    rgb[b > 0] = (220, 40, 40)
    rgb[(a > 0) & (b > 0)] = (0, 0, 0)
    scale = max(1, int(round(700 / max(shape))))
    im = Image.fromarray(np.flipud(rgb.transpose(1, 0, 2)))        # x vers la droite, y vers le haut
    im.resize((shape[0] * scale, shape[1] * scale), Image.NEAREST).save(path)


def alignment_dir(scan):
    """Dossier du scan (celui de scan.json) si le dossier lu est keyframes/, sinon le dossier lu lui-même."""
    root = scan["root"]
    return root.parent if root.name == "keyframes" and (root.parent / "scan.json").exists() else root


def register(ref, scan, a, o3d):
    cands = coarse_candidates(ref, scan, a.cell)
    for c in cands:
        print(f"  candidat {math.degrees(c['yaw']):7.1f} deg : translation ({c['T'][0, 3]:.2f}, {c['T'][1, 3]:.2f}, "
              f"{c['T'][2, 3]:.2f}) m, coïncidence {100 * c['score']:.0f} %")
    method = "prior"
    if o3d is None:
        print("  Open3D absent (pip install open3d) : pas d'affinage ICP, précision d'une cellule")
        for c in cands:
            c["fitness"], c["rmse"] = fitness_numpy(ref, apply(c["T"], scan["points"]), a.voxel), None
    else:
        src, dst = to_pcd(o3d, scan["points"], scan["normals"]), to_pcd(o3d, ref["points"], ref["normals"])
        if a.use_global:
            T0, f0 = global_registration(o3d, src, dst, a.voxel)
            T0 = project_to_vertical(T0)
            print(f"  RANSAC global : {math.degrees(yaw_of(T0[:3, :3])):.1f} deg, recouvrement {100 * f0:.0f} %")
            cands.append(dict(T=T0, yaw=yaw_of(T0[:3, :3]), score=f0))
        for c in cands:
            c["T"], c["fitness"], c["rmse"] = icp(o3d, src, dst, c["T"], a.voxel)
            c["T"] = project_to_vertical(c["T"])
        method = "prior+icp" if not a.use_global else "prior+global+icp"
    best = max(cands, key=lambda c: (c["fitness"], c["score"]))
    T = best["T"]
    P = apply(T, scan["points"])
    N = scan["normals"] @ T[:3, :3].T
    folder = alignment_dir(scan)
    info = {
        "reference": str(ref["path"]),
        "scan": str(scan["path"]),
        "transform_arkit": [[round(float(v), 6) for v in row] for row in to_arkit(T)],
        "transform_engine": [[round(float(v), 6) for v in row] for row in T],
        "yaw_deg": round(math.degrees(yaw_of(T[:3, :3])), 3),
        "translation": [round(float(v), 4) for v in T[:3, 3]],
        "fitness": round(best["fitness"], 4),
        "rmse": None if best["rmse"] is None else round(best["rmse"], 4),
        "method": method,
        "voxel": a.voxel,
        "points": [int(len(ref["points"])), int(len(scan["points"]))],
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (folder / "alignment.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    write_preview(folder / "alignment.png", ref, P, N, a.cell)
    rmse = "" if best["rmse"] is None else f", écart {100 * best['rmse']:.1f} cm"
    print(f"  retenu : {info['yaw_deg']:.1f} deg, translation {info['translation']} m, recouvrement {100 * best['fitness']:.0f} %{rmse}")
    if best["fitness"] < a.min_fitness:
        print(f"  ATTENTION : recouvrement faible (< {100 * a.min_fitness:.0f} %) : vérifier {folder / 'alignment.png'}"
              " ; essayer --global, ou scanner une zone commune plus grande")
    print(f"  -> {folder / 'alignment.json'}")
    return info


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Recalage de scans LiDAR dans le repère du premier (alignment.json)")
    ap.add_argument("reference", type=pathlib.Path, help="scan de référence (dossier keyframes, dossier du scan ou zip)")
    ap.add_argument("scans", type=pathlib.Path, nargs="+", help="scans à recaler sur la référence")
    ap.add_argument("--voxel", type=float, default=0.05, help="maille du nuage réduit (m), défaut 0.05")
    ap.add_argument("--cell", type=float, default=0.10, help="cellule des vues de dessus (m), défaut 0.10")
    ap.add_argument("--stride", type=int, default=4, help="un pixel de profondeur sur stride (défaut 4)")
    ap.add_argument("--min-conf", type=int, default=1, help="confiance LiDAR minimale 0, 1 ou 2 (défaut 1)")
    ap.add_argument("--max-depth", type=float, default=5.0, help="portée maximale gardée (m)")
    ap.add_argument("--move", type=float, default=0.05, help="SpatialCapture : déplacement (m) entre images gardées")
    ap.add_argument("--turn", type=float, default=5.0, help="SpatialCapture : rotation (degrés) entre images gardées")
    ap.add_argument("--global", dest="use_global", action="store_true",
                    help="ajoute un candidat sans a priori (FPFH + RANSAC, Open3D requis)")
    ap.add_argument("--min-fitness", type=float, default=0.3, help="recouvrement en deçà duquel un avertissement est émis")
    a = ap.parse_args()
    work = pathlib.Path(tempfile.mkdtemp(prefix="register_"))
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    o3d = open3d_or_none()
    print(f"Open3D {o3d.__version__}" if o3d else "Open3D absent")
    ref = load_cloud(a.reference, a, work)
    for p in a.scans:
        scan = load_cloud(p, a, work)
        print(f"recalage de {p.name} sur {a.reference.name} :")
        register(ref, scan, a, o3d)


if __name__ == "__main__":
    main()
