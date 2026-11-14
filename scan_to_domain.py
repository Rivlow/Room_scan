"""
scan_to_domain.py : export LiDAR « keyframes » d'iPhone  ->  géométrie de Physical_engine.

Entrée, détectée automatiquement :
  - le dossier keyframes/ d'une appli de scan LiDAR (RoomScan, Polycam : images/, depth/, confidence/, cameras/ :
    une pose ARKit, la focale et une carte de profondeur en millimètres par image clé), son dossier parent, ou
    l'archive .zip partagée par l'application ;
  - ou une prise SpatialCapture extraite du zip (manifest.json, frames.jsonl, depth.f32, confidence.u8 : vidéo
    LiDAR, une carte de profondeur en mètres et une pose ARKit par image ; seules les images où le téléphone a bougé
    de --move mètres ou tourné de --turn degrés sont gardées).

Sortie (dossier --out) :
    solids.npy      uint8, forme (nx, ny, nz), indices (i, j, k) = (x, y, z) : 0 air, 1 solide
    <nom>.json      cas Physical_engine (grille, murs, atmosphère au repos, "solids": {"file": "solids.npy"})
    colors.npy      uint8, forme (nx, ny, nz, 4) : couleur sRGB des mailles solides tirée des photos (alpha 255 mesurée,
                    128 prolongée d'une surface voisine, 0 air) ; absent si l'export n'a pas de photo
    apercu_*.png    vue de dessus des meubles et deux coupes verticales
    voxels.ply      centres des mailles solides en contact avec l'air (à ouvrir dans CloudCompare ou Blender)
    rapport.txt     dimensions, orientation, sol, plafond, nombre de mailles
    "openings"      portes et fenêtres reconnues sur le maillage ARKit (mesh.ply), voir detect_openings.py ;
                    apercu_ouvertures.png les montre vues depuis la pièce

Méthode :
  1. chaque pixel fiable de chaque carte de profondeur est projeté dans le repère du monde (pose ARKit, échelle
     métrique du LiDAR, axe vertical donné par la gravité) ;
  2. l'orientation horizontale est choisie pour aligner les murs sur les axes x et y (histogramme des normales des
     surfaces verticales, modulo 90 degrés) ;
  3. le sol est le plan horizontal le plus bas tourné vers le haut ; le plafond, le plus haut tourné vers le bas
     (sinon hauteur estimée par RoomScan dans scan.json, ou --height) ;
  4. « sculpture de l'espace libre » : chaque rayon LiDAR va de la caméra à la surface touchée ; les mailles qu'il
     traverse sont de l'air observé, la maille où il s'arrête est une surface ;
  5. une maille est solide si assez de rayons s'y arrêtent ; les mailles jamais observées sont de l'air (ou du
     solide avec --unknown solid) ; une dalle solide sous le sol et au-dessus du plafond ferme le domaine ; les
     poches d'air isolées de la zone où la caméra est passée sont rebouchées.

Usage :
    python scan_to_domain.py <dossier keyframes> --out <dossier> [--dx 0.05] [--height 2.50] [--name appartement]
"""
import argparse
import atexit
import json
import math
import pathlib
import shutil
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage


# ---------------------------------------------------------------------------------------------------------------
# lecture du scan

def to_engine(v):
    """Repère ARKit (x droite, y haut, z vers l'arrière) -> repère du moteur (x, y horizontaux, z haut)."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def frame_points(depth, rank, T, fx, fy, cx, cy, min_conf, max_depth, stride, rgb=None):
    """Une carte de profondeur (m) et sa confiance (rang 0/1/2), pose T caméra -> monde (3x4, ARKit), focales à la
    résolution de la carte -> origine, points et normales dans le repère du moteur. Avec rgb (image PIL alignée sur
    la carte, même cadrage, résolution quelconque) : couleur de chaque point (moyenne des pixels couleur couverts par
    le pixel de profondeur) et poids de visée |cos| entre la normale et le rayon (une surface vue de face compte
    plus qu'une surface rasée)."""
    depth = depth.astype(np.float64)
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    # caméra ARKit : regarde vers -z, y vers le haut de l'image
    Pc = np.stack([(u - cx) / fx * depth, -(v - cy) / fy * depth, -depth], axis=-1)
    Pw = Pc @ T[:, :3].T + T[:, 3]
    # normales des surfaces par différences centrées dans l'image (repère monde)
    du = np.zeros_like(Pw)
    dv = np.zeros_like(Pw)
    du[:, 1:-1] = Pw[:, 2:] - Pw[:, :-2]
    dv[1:-1, :] = Pw[2:, :] - Pw[:-2, :]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    # pixels gardés : profondeur valide, confiance suffisante, pas de saut de profondeur (pixels volants des bords)
    with np.errstate(invalid="ignore"):
        dd = np.zeros_like(depth)
        dd[1:-1, 1:-1] = np.maximum.reduce([abs(depth[1:-1, 2:] - depth[1:-1, :-2]), abs(depth[2:, 1:-1] - depth[:-2, 1:-1])])
        ok = np.isfinite(depth) & (depth > 0.15) & (depth < max_depth) & (rank >= min_conf) \
            & (dd < 0.05 + 0.03 * depth) & (nn[..., 0] > 0)
    sel = np.zeros_like(ok)
    sel[::stride, ::stride] = True
    ok &= sel
    # normale orientée vers la caméra
    o = T[:, 3]
    flip = np.einsum("hwc,hwc->hw", n, o - Pw) < 0
    n[flip] *= -1
    out = {"origin": to_engine(o), "points": to_engine(Pw[ok]), "normals": to_engine(n[ok])}
    if rgb is not None:
        col = np.asarray(rgb.convert("RGB").resize((w, h), Image.BOX))
        ray = o - Pw[ok]
        cosv = np.abs(np.einsum("nc,nc->n", n[ok], ray)) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9)
        out["colors"], out["weights"] = col[ok], cosv
    return out


def confidence_rank(conf):
    """Carte de confiance ARKit -> rang 0 (faible), 1 (moyenne), 2 (haute). RoomScan écrit le rang tel quel ; d'autres
    applis l'étalent sur 0..255 (0 / 54 / 255 pour Polycam)."""
    c32 = conf.astype(np.int32)
    return c32 if c32.max() <= 2 else np.where(c32 < 20, 0, np.where(c32 < 200, 1, 2))


def confidence_counts(root):
    """Nombre de pixels de profondeur de confiance faible, moyenne et haute dans tout le scan ; None si le scan n'a
    pas de carte de confiance (toutes ses mesures comptent alors comme hautes)."""
    counts = np.zeros(3, np.int64)
    if is_spatialcapture(root):
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        cpath = root / man.get("confidence", "confidence.u8")
        if not cpath.exists() or cpath.stat().st_size == 0:
            return None
        data = np.memmap(cpath, dtype=np.uint8, mode="r")
        for a in range(0, len(data), 1 << 24):                  # par blocs : le fichier peut faire des Go
            counts += np.bincount(np.minimum(data[a:a + (1 << 24)], 2), minlength=3)
        return counts
    files = sorted((root / "confidence").glob("*.png"))
    if not files:
        return None
    for f in files:
        counts += np.bincount(confidence_rank(np.asarray(Image.open(f))).ravel(), minlength=3)[:3]
    return counts


def load_keyframes(root, min_conf, max_depth, stride, color=True):
    """Dossier keyframes/ (RoomScan, Polycam) : cameras/*.json, depth/*.png (mm), confidence/*.png (et les photos
    images/*.jpg pour la couleur des mailles, sauf color=False)."""
    frames = []
    for cf in sorted((root / "cameras").glob("*.json")):
        cam = json.loads(cf.read_text())
        ts = cf.stem
        dpath, cpath = root / "depth" / f"{ts}.png", root / "confidence" / f"{ts}.png"
        if not dpath.exists():
            continue
        T = np.array([[cam[f"t_{r}{c}"] for c in range(4)] for r in range(3)], dtype=np.float64)    # caméra -> monde
        depth = np.asarray(Image.open(dpath)).astype(np.float64) / 1000.0                          # mm -> m
        h, w = depth.shape
        rank = confidence_rank(np.asarray(Image.open(cpath))) if cpath.exists() else np.full((h, w), 2, np.int32)
        sx, sy = w / cam["width"], h / cam["height"]
        ipath = root / "images" / f"{ts}.jpg"
        rgb = Image.open(ipath) if color and ipath.exists() else None
        frames.append(frame_points(depth, rank, T, cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy,
                                   min_conf, max_depth, stride, rgb))
    return frames


def is_spatialcapture(root):
    m = root / "manifest.json"
    if not m.exists():
        return False
    try:
        return json.loads(m.read_text(encoding="utf-8")).get("app") == "SpatialCapture"
    except (OSError, ValueError):
        return False


def select_moving(poses, move, turn):
    """Indices des poses gardées : la première, puis chaque pose qui s'écarte de la dernière gardée de plus de move
    (m) en position ou de turn (degrés) en direction de visée. Les images d'une vidéo à 30 images/s sont presque
    identiques : sans tri, elles répètent les mêmes rayons et coûtent du temps sans rien ajouter."""
    keep, last = [], None
    cos_t = math.cos(math.radians(turn))
    for n, T in enumerate(poses):
        if last is not None:
            d = np.linalg.norm(T[:, 3] - last[:, 3])
            c = float(np.dot(T[:, 2], last[:, 2]))             # axes de visée (-z caméra)
            if d < move and c > cos_t:
                continue
        keep.append(n)
        last = T
    return keep


def load_spatialcapture(root, min_conf, max_depth, stride, move, turn, color=True):
    """Prise SpatialCapture (zip extrait) : manifest.json, frames.jsonl (pose par ligne, matrices en lignes,
    intrinsèques de l'image couleur), depth.f32 (float32 m, cartes concaténées sans remplissage), confidence.u8
    (0/1/2). Seules les images à suivi normal et avec profondeur sont lues, triées par mouvement (select_moving)."""
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if man.get("depthCaptureMode", "sceneDepth") != "sceneDepth":
        sys.exit("prise SpatialCapture sans LiDAR (depthCaptureMode = none) : aucune géométrie mesurée")
    if man.get("state") not in (None, "complete"):
        print(f"attention : prise dans l'état « {man.get('state')} », pas complète")
    rows = []
    for line in (root / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            f = json.loads(line)
        except ValueError:
            continue                                            # dernière ligne coupée d'une prise interrompue
        if f.get("trackingState") == "normal" and "depthByteOffset" in f:
            rows.append(f)
    poses = [np.array(f["cameraToWorld"], dtype=np.float64)[:3] for f in rows]
    # images triées par mouvement, plus toutes celles qui ont une photo couleur (rgb/*.jpg, ~2 par seconde)
    with_rgb = {n for n, f in enumerate(rows) if f.get("rgbKeyframe") and (root / f["rgbKeyframe"]).exists()}
    keep = sorted(set(select_moving(poses, move, turn)) | with_rgb)
    depth_all = np.memmap(root / man.get("depth", "depth.f32"), dtype="<f4", mode="r")
    cpath = root / man.get("confidence", "confidence.u8")
    conf_all = np.memmap(cpath, dtype=np.uint8, mode="r") if cpath.exists() and cpath.stat().st_size else None
    frames = []
    for n in keep:
        f = rows[n]
        w, h = f["depthWidth"], f["depthHeight"]
        o = f["depthByteOffset"] // 4
        depth = np.array(depth_all[o:o + w * h]).reshape(h, w)
        if conf_all is not None and "confidenceByteOffset" in f:
            c = f["confidenceByteOffset"]
            rank = np.array(conf_all[c:c + w * h]).reshape(h, w).astype(np.int32)
        else:
            rank = np.full((h, w), 2, np.int32)
        K = f["intrinsics"]
        sx, sy = w / f["imageWidth"], h / f["imageHeight"]
        rgb = Image.open(root / f["rgbKeyframe"]) if color and n in with_rgb else None
        frames.append(frame_points(depth, rank, poses[n], K[0][0] * sx, K[1][1] * sy, K[0][2] * sx, K[1][2] * sy,
                                   min_conf, max_depth, stride, rgb))
    print(f"SpatialCapture : {len(rows)} images à suivi normal, {len(frames)} gardées "
          f"(déplacement > {move} m ou rotation > {turn} degrés, ou photo couleur : {len(with_rgb)})")
    return frames


def is_scan_root(d):
    return is_spatialcapture(d) or (d / "depth").is_dir()


def resolve_source(path, out):
    """Dossier du scan (ou un dossier parent, comme l'export RoomScan qui contient keyframes/) ou archive .zip ->
    dossier qui contient les données. Une archive est extraite dans un dossier temporaire de --out, effacé à la fin."""
    if path.suffix.lower() == ".zip":
        work = pathlib.Path(tempfile.mkdtemp(prefix="source_", dir=out))
        atexit.register(shutil.rmtree, work, ignore_errors=True)
        print(f"extraction de {path.name}")
        with zipfile.ZipFile(path) as z:
            z.extractall(work)
        path = work
    if not path.is_dir():
        sys.exit(f"scan introuvable : {path}")
    if is_scan_root(path):
        return path
    for d in sorted((p for p in path.rglob("*") if p.is_dir()), key=lambda p: len(p.parts)):
        if is_scan_root(d):
            return d
    sys.exit(f"aucun scan reconnu dans {path} (attendu : keyframes/ avec depth/, ou manifest.json de SpatialCapture)")


def load_frames(root, min_conf, max_depth, stride, move=0.05, turn=5.0, color=True):
    """Lit un dossier keyframes/ (RoomScan, Polycam) ou une prise SpatialCapture, selon ce qu'il contient."""
    if is_spatialcapture(root):
        frames = load_spatialcapture(root, min_conf, max_depth, stride, move, turn, color)
    else:
        frames = load_keyframes(root, min_conf, max_depth, stride, color)
    if not frames:
        sys.exit(f"aucune image lisible dans {root}")
    return frames


def read_scan_meta(keyframes):
    """scan.json écrit par RoomScan à côté du dossier keyframes (vide s'il n'existe pas)."""
    for p in (keyframes.parent / "scan.json", keyframes / "scan.json"):
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
    return {}


# ---------------------------------------------------------------------------------------------------------------
# plusieurs scans : recalage écrit par register_scans.py

E_ENGINE = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # p_moteur = E p_arkit (to_engine)


def read_alignment(root):
    """alignment.json de register_scans.py (dans le dossier lu ou son parent) -> (matrice 4x4 ARKit -> ARKit de la
    référence, matrice 4x4 équivalente dans le repère du moteur, infos) ; None s'il n'y en a pas."""
    for p in (root / "alignment.json", root.parent / "alignment.json"):
        if p.exists():
            info = json.loads(p.read_text(encoding="utf-8"))
            M = np.array(info["transform_arkit"], float)
            T = np.eye(4)
            T[:3, :3] = E_ENGINE @ M[:3, :3] @ E_ENGINE.T
            T[:3, 3] = E_ENGINE @ M[:3, 3]
            return M, T, info
    return None


def apply_alignment(frames, T):
    """Applique une transformation 4x4 (repère du moteur) aux points, normales et origines des images."""
    R, t = T[:3, :3], T[:3, 3]
    for f in frames:
        f["points"] = f["points"] @ R.T + t
        f["normals"] = f["normals"] @ R.T
        f["origin"] = R @ f["origin"] + t
    return frames


def merge_meshes(items, out_path):
    """Un seul mesh.ply (format RoomScan, repère ARKit de la référence) à partir de (chemin, matrice 4x4 ARKit) ; la
    classe par face est conservée. None si aucun maillage."""
    from detect_openings import read_ply, write_ply_mesh
    Vs, Fs, base = [], [], 0
    for path, M in items:
        if path is None:
            continue
        V, F = read_ply(path)
        V = V.copy()
        V[:, :3] = V[:, :3] @ M[:3, :3].T + M[:3, 3]
        V[:, 3:6] = V[:, 3:6] @ M[:3, :3].T
        F = F.copy()
        F["i"] += base
        base += len(V)
        Vs.append(V)
        Fs.append(F)
    if not Vs:
        return None
    write_ply_mesh(out_path, np.concatenate(Vs), np.concatenate(Fs))
    return out_path


# ---------------------------------------------------------------------------------------------------------------
# orientation, sol, plafond

def dominant_yaw(normals):
    """Angle (rad) qui aligne les murs sur les axes : histogramme pondéré de l'azimut des normales horizontales mod 90°."""
    horiz = np.abs(normals[:, 2]) < 0.3
    nx, ny = normals[horiz, 0], normals[horiz, 1]
    wgt = np.hypot(nx, ny)
    ang = np.mod(np.arctan2(ny, nx), np.pi / 2)
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi / 2), weights=wgt)
    hist = ndimage.gaussian_filter1d(hist, 2, mode="wrap")
    return 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])


def horizontal_planes(z, normals, up, bin_m=0.02):
    """Pics de l'histogramme des hauteurs des surfaces tournées vers le haut (up = +1) ou vers le bas (up = -1)."""
    m = normals[:, 2] * up > 0.9
    if m.sum() < 100:
        return []
    zz = z[m]
    lo, hi = np.percentile(zz, 0.1) - 5 * bin_m, np.percentile(zz, 99.9) + 5 * bin_m        # marge : un pic au bord reste un pic
    nb = max(int((hi - lo) / bin_m), 1)
    hist, edges = np.histogram(zz, bins=nb, range=(lo, hi))
    hist = ndimage.gaussian_filter1d(hist.astype(float), 1)
    peaks = [i for i in range(1, nb - 1) if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0.05 * hist.max()]
    return [(float(0.5 * (edges[i] + edges[i + 1])), float(hist[i])) for i in peaks]


def room_frame(frames, a, root):
    """Repère du domaine : orientation des murs, sol, plafond et emprise horizontale, détectés ou imposés (--yaw,
    --floor, --ceiling, --crop). Tourne les points et les origines des images (frames) autour de la verticale ; les
    coordonnées rendues sont celles de ce repère tourné (m)."""
    pts = np.concatenate([f["points"] for f in frames])
    nrm = np.concatenate([f["normals"] for f in frames])
    yaw = math.radians(a.yaw) if a.yaw is not None else dominant_yaw(nrm)
    c, s = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for f in frames:
        f["points"] = f["points"] @ Rz.T
        f["origin"] = Rz @ f["origin"]
    pts, nrm = pts @ Rz.T, nrm @ Rz.T

    floors = horizontal_planes(pts[:, 2], nrm, +1)
    if a.floor is not None:
        floor = a.floor
    elif floors:
        floor = min(z for z, _ in floors)
    else:
        sys.exit("sol introuvable : le scan doit montrer le sol (ou --floor)")
    ceil_cands = [z for z, w in horizontal_planes(pts[:, 2], nrm, -1) if z > floor + 1.8]
    ceiling_seen = bool(ceil_cands)
    if a.ceiling is not None:
        ceiling, height_src = a.ceiling, "imposé"
    elif ceil_cands:
        ceiling, height_src = max(ceil_cands), "scanné"
    else:
        h, height_src = a.height, "--height"
        meta = read_scan_meta(root)
        if h is None and meta.get("floorHeight") is not None and meta.get("ceilingHeight") is not None:
            h = float(meta["ceilingHeight"]) - float(meta["floorHeight"])
            height_src = "maillage ARKit (scan.json)" if h > 1.8 else height_src
            h = h if h > 1.8 else None
        if h is None:
            h, height_src = 2.50, "défaut 2.50 m"
        ceiling = floor + h
    if ceiling <= floor:
        sys.exit(f"plafond ({ceiling:.3f} m) sous le sol ({floor:.3f} m)")

    if a.crop is not None:
        lo_xy, hi_xy = np.array([a.crop[0], a.crop[2]]), np.array([a.crop[1], a.crop[3]])
        if np.any(hi_xy <= lo_xy):
            sys.exit(f"découpe vide : --crop {a.crop}")
    else:
        lo_xy = np.percentile(pts[:, :2], a.margin, axis=0)
        hi_xy = np.percentile(pts[:, :2], 100 - a.margin, axis=0)
    return dict(yaw=yaw, Rz=Rz, floor=floor, ceiling=ceiling, ceiling_seen=ceiling_seen, height_src=height_src,
                floors=floors, lo_xy=lo_xy, hi_xy=hi_xy, n_points=len(pts))


def write_preview(out, root, source, frames, fr, mesh=None):
    """Aperçu pour la préparation du scan dans Physical_engine, sans voxéliser : preview.json (repère détecté,
    confiance) et preview_mesh.bin, le maillage ARKit de RoomScan (`mesh`, sinon celui trouvé à côté du scan) dans le
    repère tourné :
        char[4] "PMSH", uint32 version (1), uint32 nv, uint32 nt
        nv sommets de 24 octets : float x, y, z, nx, ny, nz (normale tournée vers l'air)
        nt triangles de 3 uint32, puis nt uint8 : classe ARKit (0 aucune, 1 mur, 2 sol, 3 plafond, 4 table, 5 siège,
        6 fenêtre, 7 porte)"""
    from detect_openings import find_mesh, read_ply
    mesh = mesh or find_mesh(root, source if source.is_dir() else None)
    if mesh is not None:
        V, F = read_ply(mesh)
        P = to_engine(V[:, :3]) @ fr["Rz"].T
        N = to_engine(V[:, 3:6]) @ fr["Rz"].T
        with open(out / "preview_mesh.bin", "wb") as fh:
            fh.write(b"PMSH" + np.array([1, len(P), len(F)], "<u4").tobytes())
            fh.write(np.concatenate([P, N], 1).astype("<f4").tobytes())
            fh.write(np.ascontiguousarray(F["i"]).astype("<u4").tobytes())
            fh.write(np.ascontiguousarray(F["c"]).astype(np.uint8).tobytes())
    counts = confidence_counts(root)
    info = {
        "source": str(source),
        "frames": len(frames),
        "points": fr["n_points"],
        "yaw_deg": round(math.degrees(fr["yaw"]), 4),
        "floor": round(float(fr["floor"]), 4),
        "ceiling": round(float(fr["ceiling"]), 4),
        "ceiling_scanned": fr["ceiling_seen"],
        "height_source": fr["height_src"],
        "planes": [round(z - fr["floor"], 3) for z, _ in fr["floors"]],
        "crop": [round(float(v), 4) for v in (fr["lo_xy"][0], fr["hi_xy"][0], fr["lo_xy"][1], fr["hi_xy"][1])],
        "confidence": None if counts is None else [int(n) for n in counts],
        "mesh": "preview_mesh.bin" if mesh is not None else None,
    }
    (out / "preview.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"aperçu : orientation {info['yaw_deg']:.1f} deg, hauteur {fr['ceiling'] - fr['floor']:.2f} m "
          f"({fr['height_src']}), maillage {'oui' if mesh is not None else 'absent'}")


# ---------------------------------------------------------------------------------------------------------------
# sculpture de l'espace libre

def carve(frames, lo, shape, dx):
    n_cells = int(np.prod(shape))
    free = np.zeros(n_cells, np.int64)
    hits = np.zeros(n_cells, np.int64)
    shp = np.array(shape)

    def flat(p):
        ijk = np.floor((p - lo) / dx).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < shp), axis=-1)
        return (ijk[..., 0] * shape[1] + ijk[..., 1]) * shape[2] + ijk[..., 2], inside

    step = 0.5 * dx
    for f in frames:
        o, pts = f["origin"], f["points"]
        idx, inside = flat(pts)
        hits += np.bincount(idx[inside], minlength=n_cells)
        d = pts - o
        L = np.linalg.norm(d, axis=1)
        d /= L[:, None]
        for a in range(0, len(pts), 20000):
            dc, Lc = d[a:a + 20000], L[a:a + 20000]
            ns = int(math.ceil(Lc.max() / step))
            t = (np.arange(ns) + 0.5) * step
            p = o + dc[:, None, :] * t[None, :, None]
            keep = t[None, :] < (Lc[:, None] - 0.75 * dx)        # on s'arrête une maille avant la surface
            fi, ins = flat(p)
            free += np.bincount(fi[keep & ins], minlength=n_cells)
    return free.reshape(shape), hits.reshape(shape)


def paint(frames, lo, shape, dx):
    """Couleur moyenne (sRGB 0..255, pondérée par le poids de visée) des points LiDAR colorés tombant dans chaque
    maille, et somme des poids (0 : maille sans échantillon)."""
    n_cells = int(np.prod(shape))
    acc = np.zeros((n_cells, 3))
    wsum = np.zeros(n_cells)
    shp = np.array(shape)
    for f in frames:
        if "colors" not in f:
            continue
        ijk = np.floor((f["points"] - lo) / dx).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < shp), axis=-1)
        idx = ((ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2])[inside]
        wgt = f["weights"][inside]
        wsum += np.bincount(idx, weights=wgt, minlength=n_cells)
        for c in range(3):
            acc[:, c] += np.bincount(idx, weights=wgt * f["colors"][inside, c], minlength=n_cells)
    mean = acc / np.maximum(wsum, 1e-12)[:, None]
    return mean.reshape(tuple(shape) + (3,)), wsum.reshape(shape)


def solid_colors(solid, mean, wsum):
    """Couleur RGBA (uint8) de chaque maille solide : celle de la maille échantillonnée la plus proche (la surface
    mesurée tombe dans la maille solide ou dans l'air juste devant). Alpha 255 : échantillon à une maille au plus ;
    128 : couleur prolongée de plus loin (surface jamais vue en couleur) ; 0 : air."""
    sampled = wsum > 0
    rgba = np.zeros(solid.shape + (4,), np.uint8)
    if not sampled.any():
        return rgba, 0.0
    dist, ind = ndimage.distance_transform_edt(~sampled, return_indices=True)
    near = mean[ind[0], ind[1], ind[2]]
    rgba[..., :3] = np.clip(np.round(near), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(dist <= 1.0, 255, 128)
    rgba[~solid] = 0
    # part des faces solides exposées à l'air dont la couleur est mesurée
    air = np.pad(~solid, 1, constant_values=False)
    touches = np.zeros_like(solid)
    for ax in range(3):
        for sgn in (-1, 1):
            touches |= np.roll(air, sgn, axis=ax)[1:-1, 1:-1, 1:-1]
    exposed = solid & touches
    measured = float((rgba[..., 3][exposed] == 255).mean()) if exposed.any() else 0.0
    return rgba, measured


# ---------------------------------------------------------------------------------------------------------------
# sorties

def write_ply(path, solid, lo, dx, rgba=None):
    air = ~solid
    pad = np.pad(air, 1, constant_values=False)
    touches = np.zeros_like(solid)
    for ax in range(3):
        for s in (-1, 1):
            touches |= np.roll(pad, s, axis=ax)[1:-1, 1:-1, 1:-1]
    surf = solid & touches
    ijk = np.argwhere(surf)
    xyz = lo + (ijk + 0.5) * dx
    z = xyz[:, 2]
    t = (z - z.min()) / max(np.ptp(z), 1e-9)
    rgb = np.stack([255 * t, 80 + 120 * (1 - abs(2 * t - 1)), 255 * (1 - t)], 1).astype(np.uint8)
    if rgba is not None and rgba[..., 3].any():
        rgb = rgba[ijk[:, 0], ijk[:, 1], ijk[:, 2], :3]           # couleurs du scan
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, rgb):
            fh.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
    return len(xyz)


def save_previews(out, solid, dx, k_floor, k_ceil):
    scale = max(1, int(round(600 / max(solid.shape[0], solid.shape[1]))))
    # dessus : hauteur du plus haut solide entre sol et plafond (meubles), 0 = sol libre
    inner = solid[:, :, k_floor + 1:k_ceil]
    top = np.where(inner.any(2), inner.shape[2] - np.argmax(inner[:, :, ::-1], axis=2), 0) if inner.shape[2] else np.zeros(solid.shape[:2])
    img = np.where(top > 0, 60 + 195 * top / max(top.max(), 1), 0).astype(np.uint8)        # noir = sol libre, clair = haut
    Image.fromarray(np.flipud(img.T)).resize((img.shape[0] * scale, img.shape[1] * scale), Image.NEAREST).save(out / "apercu_dessus.png")
    for name, sec in (("apercu_coupe_x.png", solid[solid.shape[0] // 2, :, :]), ("apercu_coupe_y.png", solid[:, solid.shape[1] // 2, :])):
        im = np.where(sec, 70, 225).astype(np.uint8)
        Image.fromarray(np.flipud(im.T)).resize((im.shape[0] * scale, im.shape[1] * scale), Image.NEAREST).save(out / name)


# ---------------------------------------------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Export LiDAR keyframes -> mailles solides de Physical_engine")
    ap.add_argument("keyframes", type=pathlib.Path, nargs="+",
                    help="dossier keyframes/ (RoomScan, Polycam) ou prise SpatialCapture ; plusieurs scans : le premier "
                         "est la référence, les autres sont recalés par leur alignment.json (register_scans.py)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--name", default="appartement")
    ap.add_argument("--dx", type=float, default=0.05, help="taille des mailles (m), défaut 0.05")
    ap.add_argument("--height", type=float, default=None,
                    help="hauteur sous plafond si le plafond n'est pas reconnu (m) ; défaut : scan.json de RoomScan, sinon 2.50")
    ap.add_argument("--min-conf", type=int, default=1, help="confiance LiDAR minimale 0, 1 ou 2 (défaut 1)")
    ap.add_argument("--max-depth", type=float, default=5.0, help="portée maximale gardée (m)")
    ap.add_argument("--stride", type=int, default=1, help="un pixel de profondeur sur stride (accélère)")
    ap.add_argument("--min-hits", type=int, default=2, help="rayons arrêtés minimum pour une maille solide")
    ap.add_argument("--unknown", choices=["air", "solid"], default="air", help="mailles jamais observées")
    ap.add_argument("--close", type=int, default=1, help="fermeture morphologique des solides (mailles), 0 = aucune")
    ap.add_argument("--margin", type=float, default=0.5, help="percentile (%%) des points écartés aux bords (bruit)")
    ap.add_argument("--move", type=float, default=0.05,
                    help="SpatialCapture : déplacement (m) entre deux images gardées, défaut 0.05")
    ap.add_argument("--turn", type=float, default=5.0,
                    help="SpatialCapture : rotation (degrés) entre deux images gardées, défaut 5")
    # repère imposé (préparation du scan dans Physical_engine), dans le repère tourné de l'aperçu
    ap.add_argument("--yaw", type=float, default=None, help="orientation des murs imposée (degrés)")
    ap.add_argument("--floor", type=float, default=None, help="altitude du sol imposée (m, repère ARKit)")
    ap.add_argument("--ceiling", type=float, default=None, help="altitude du plafond imposée (m, repère ARKit)")
    ap.add_argument("--crop", type=float, nargs=4, default=None, metavar=("X0", "X1", "Y0", "Y1"),
                    help="emprise horizontale imposée (m, repère tourné)")
    ap.add_argument("--complete-walls", action="store_true",
                    help="bouche les trous de mesure des murs (plans des triangles « mur » du mesh.ply ARKit)")
    ap.add_argument("--preview", action="store_true",
                    help="écrit seulement l'aperçu (preview.json, preview_mesh.bin), sans voxéliser")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dx = a.dx
    a.sources, a.keyframes = a.keyframes, a.keyframes[0]

    from detect_openings import find_mesh
    root = resolve_source(a.keyframes, a.out)
    frames = load_frames(root, a.min_conf, a.max_depth, a.stride, a.move, a.turn)
    mesh_path = find_mesh(root, a.keyframes if a.keyframes.is_dir() else None)
    # scans supplémentaires, dans le repère ARKit de la référence (alignment.json de register_scans.py)
    meshes = [(mesh_path, np.eye(4))]
    for p in a.sources[1:]:
        r = resolve_source(p, a.out)
        al = read_alignment(r)
        if al is None:
            sys.exit(f"{p} : pas d'alignment.json ; le recaler d'abord : python register_scans.py {a.keyframes} {p}")
        M, T, info = al
        extra = apply_alignment(load_frames(r, a.min_conf, a.max_depth, a.stride, a.move, a.turn), T)
        print(f"{p.name} : {len(extra)} images clés recalées ({info.get('yaw_deg', 0):.1f} deg, recouvrement "
              f"{100 * info.get('fitness', 0):.0f} %)")
        frames += extra
        meshes.append((find_mesh(r, p if p.is_dir() else None), M))
    if len(meshes) > 1:
        mesh_path = merge_meshes(meshes, a.out / "mesh_merged.ply")
    fr = room_frame(frames, a, root)
    print(f"{len(frames)} images clés, {fr['n_points']} points LiDAR")
    if a.preview:
        write_preview(a.out, root, a.keyframes, frames, fr, mesh_path)
        return
    yaw, floor, ceiling, lo_xy, hi_xy = fr["yaw"], fr["floor"], fr["ceiling"], fr["lo_xy"], fr["hi_xy"]
    ceiling_seen, height_src, floors = fr["ceiling_seen"], fr["height_src"], fr["floors"]
    cams = np.array([f["origin"] for f in frames])
    n_points = fr["n_points"]

    # boîte : sol en z = 0, une dalle de mailles sous le sol et au-dessus du plafond
    nz_room = int(round((ceiling - floor) / dx))
    nx = int(math.ceil((hi_xy[0] - lo_xy[0]) / dx))
    ny = int(math.ceil((hi_xy[1] - lo_xy[1]) / dx))
    nz = nz_room + 2
    lo = np.array([lo_xy[0], lo_xy[1], floor - dx])
    print(f"grille {nx} x {ny} x {nz} ({nx * ny * nz} mailles), dx = {dx} m")

    free, hits = carve(frames, lo, (nx, ny, nz), dx)

    # classification
    solid = (hits >= a.min_hits) & (hits >= 0.25 * free)
    observed = (free > 0) | (hits > 0)
    if a.unknown == "solid":
        solid |= ~observed
    if a.close > 0:
        solid = ndimage.binary_closing(solid, structure=np.ones((3, 3, 3), bool), iterations=a.close, border_value=0)
    walls_line = "murs complétés         non (--complete-walls)"
    if a.complete_walls:
        from complete_walls import complete_walls
        if mesh_path is None:
            walls_line = "murs complétés         impossible : pas de mesh.ply RoomScan"
        else:
            before = int(solid.sum())
            solid, planes = complete_walls(solid, free, mesh_path, fr["Rz"], lo, dx)
            walls_line = (f"murs complétés         {int(solid.sum()) - before} mailles bouchées sur {len(planes)} plans "
                          f"({', '.join(f'{ax} = {pos:.2f} m' for ax, pos in planes)})")
        print(walls_line)
    # petits amas isolés (pixels volants) : retirés
    lab, nlab = ndimage.label(solid)
    if nlab:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
        solid &= np.isin(lab, 1 + np.flatnonzero(sizes >= 4))
    solid[:, :, 0] = True                  # dalle sous le sol
    solid[:, :, nz - 1] = True             # dalle au-dessus du plafond
    # poches d'air non reliées à la zone parcourue par la caméra : rebouchées
    lab, nlab = ndimage.label(~solid)
    cam_ijk = np.clip(np.floor((cams - lo) / dx).astype(int), 0, np.array([nx, ny, nz]) - 1)
    keep = {lab[tuple(ijk)] for ijk in cam_ijk} - {0}
    if not keep:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
        keep = {1 + int(np.argmax(sizes))}
    air = np.isin(lab, list(keep))
    solid = ~air

    kinds = solid.astype(np.uint8)
    np.save(a.out / "solids.npy", kinds)
    # couleurs des photos : une couleur RGBA par maille solide (colors.npy), lue par le moteur pour l'affichage
    mean, wsum = paint(frames, lo, (nx, ny, nz), dx)
    rgba, measured = solid_colors(solid, mean, wsum)
    has_color = bool(rgba[..., 3].any())
    if has_color:
        np.save(a.out / "colors.npy", rgba)
    n_surf = write_ply(a.out / "voxels.ply", solid, lo - np.array([lo[0], lo[1], floor]), dx, rgba if has_color else None)
    save_previews(a.out, solid, dx, 0, nz - 1)

    Lx, Ly, Lz = nx * dx, ny * dx, nz * dx
    case = {
        "label": f"Scan : {a.name}",
        "description": f"Pièce scannée au LiDAR ({len(frames)} images clés), mailles de {dx * 100:.0f} cm : "
                       "murs, sol, plafond et meubles en mailles solides, air au repos à 20 °C.",
        "grid": {"nx": nx, "ny": ny, "nz": nz, "Lx": round(Lx, 6), "Ly": round(Ly, 6), "Lz": round(Lz, 6)},
        "numerics": {"cfl": 0.5, "scheme": "ssprk2"},
        "air": {"model": "anelastic"},                 # pas d'ondes sonores : pas de temps advectif (~dx / 1 m/s)
        "bc": {"west": "wall", "east": "wall", "south": "wall", "north": "wall", "bottom": "wall", "top": "wall"},
        "atmosphere": {"type": "isentropic", "theta0": 293.15, "p0": 1e5, "rh0": 0.5, "rh_decay_height": 2000},
        "air_init": {"type": "rest"},
        "solids": {"file": "solids.npy", **({"colors": "colors.npy"} if has_color else {})},
        "display": {"field": "theta_pert", "colormap": "coolwarm", "slices": {}, "volume": "field"},
        "scan": {"source": str(a.keyframes), "dx": dx, "yaw_deg": round(math.degrees(yaw), 3),
                 "origin_arkit_floor": [round(float(v), 4) for v in (lo[0], lo[1], floor)],
                 "ceiling_height": round(ceiling - floor, 3), "ceiling_scanned": ceiling_seen},
    }
    if len(a.sources) > 1:
        case["scan"]["sources"] = [str(p) for p in a.sources]
        if mesh_path is not None:
            case["scan"]["mesh"] = mesh_path.name                 # maillages fusionnés, à côté du cas
    # portes et fenêtres du maillage ARKit (mesh.ply de RoomScan) : proposées dans « Bords et ouvertures » du moteur
    openings_line = "portes et fenêtres     pas de mesh.ply RoomScan : non détectées"
    try:
        from detect_openings import detect, save_preview
        mesh = mesh_path
        if mesh is not None:
            print("portes et fenêtres :")
            case["openings"] = detect(mesh, case, solid)
            save_preview(a.out / "apercu_ouvertures.png", case["openings"], solid, rgba if has_color else None)
            names = ", ".join(f"{o['name']} {o['width']:.2f} x {o['height']:.2f} m" for o in case["openings"])
            openings_line = f"portes et fenêtres     {names or 'aucune reconnue'} (apercu_ouvertures.png)"
    except Exception as e:
        openings_line = f"portes et fenêtres     non détectées : {e}"
    case_path = a.out / f"{a.name}.json"
    case_path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    # maillage fin pour l'affichage réaliste : facultatif, l'échec n'empêche pas la simulation
    try:
        from scan_to_mesh import build_mesh
        build_mesh(a.keyframes, case_path)
    except Exception as e:
        print(f"maillage fin non créé : {e}")

    furn = solid[:, :, 1:nz - 1]
    report = [
        f"source                 {a.keyframes}" + (f" + {len(a.sources) - 1} scan(s) recalé(s) : "
                                                    + ", ".join(str(p) for p in a.sources[1:]) if len(a.sources) > 1 else ""),
        f"images clés            {len(frames)}  ({n_points} points LiDAR)",
        f"orientation des murs   {math.degrees(yaw):.1f} deg (rotation appliquée autour de la verticale)",
        f"surfaces horizontales  {[round(z - floor, 3) for z, _ in floors]}  (m au-dessus du sol : sol, plateaux)",
        f"hauteur sous plafond   {ceiling - floor:.3f} m  ({height_src})",
        f"domaine                {Lx:.2f} x {Ly:.2f} x {Lz:.2f} m",
        f"grille                 {nx} x {ny} x {nz} = {nx * ny * nz} mailles, dx = {dx} m",
        f"air                    {int(air.sum())} mailles = {air.sum() * dx ** 3:.2f} m3",
        f"solide entre sol et plafond  {int(furn.sum())} mailles ({100 * furn.mean():.1f} %)",
        f"mailles observées      {100 * observed.mean():.1f} %  (le reste : --unknown {a.unknown})",
        walls_line,
        f"faces solides exposées {n_surf} (voxels.ply)",
        (f"couleurs               {100 * measured:.1f} % des faces exposées mesurées sur les photos, le reste prolongé "
         f"(colors.npy, {sum('colors' in f for f in frames)} images en couleur)") if has_color
        else "couleurs               aucune photo : pas de colors.npy",
        openings_line,
    ]
    (a.out / "rapport.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
