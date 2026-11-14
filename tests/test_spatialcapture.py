"""
Test du format SpatialCapture -> scan_to_domain.py, sur la même pièce synthétique que test_format.py.

On écrit une prise comme l'exporte l'application (manifest.json, frames.jsonl avec cameraToWorld en lignes et les
intrinsèques de l'image 1920x1440, depth.f32 en float32 m concaténés, confidence.u8 0/1/2), avec ce qu'une vraie
vidéo contient : chaque pose répétée (caméra quasi immobile entre deux images à 30 images/s), une image à suivi
limité dont la pose est fausse, des pixels NaN, et une dernière ligne JSON coupée. Le convertisseur doit retrouver
la même pièce, ne garder qu'une image par pose et ignorer l'image à suivi limité.

    python -m pytest tests -q        (depuis Room_scan/)
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_format import CX, CY, FX, H, W, synthetic_frames   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SX = 1920 / W                                             # intrinsèques données pour l'image couleur 1920x1440


def make_take(dst: pathlib.Path):
    dst.mkdir(parents=True)
    rows = []
    t = 0.0
    with open(dst / "depth.f32", "wb") as fd, open(dst / "confidence.u8", "wb") as fc:
        for n, (zdepth, Rwc, tw) in enumerate(synthetic_frames()):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rwc, tw
            for rep in range(3):                          # trois images quasi identiques par pose
                depth = zdepth.astype("<f4").copy()
                depth[5, 7] = np.nan                       # pixels invalides
                depth[100, 3] = np.inf
                conf = np.full((H, W), 2, np.uint8)
                conf[:, :4] = 0                            # bord de faible confiance
                Tn = T.copy()
                Tn[:3, 3] += 0.002 * rep                   # 2 mm : sous le seuil de déplacement
                tracking = "normal"
                if n == 7 and rep == 1:                    # pose fausse, à ignorer
                    tracking = "limited"
                    Tn[:3, 3] += 3.0
                rows.append({"index": len(rows), "timestamp": t, "trackingState": tracking,
                             "cameraToWorld": Tn.tolist(),
                             "intrinsics": [[FX * SX, 0, CX * SX], [0, FX * SX, CY * SX], [0, 0, 1]],
                             "imageWidth": 1920, "imageHeight": 1440, "depthWidth": W, "depthHeight": H,
                             "depthByteOffset": fd.tell(), "confidenceByteOffset": fc.tell()})
                fd.write(depth.tobytes())
                fc.write(conf.tobytes())
                t += 1 / 30
    lines = [json.dumps(r) for r in rows]
    lines.append('{"index": 99999, "timestamp": 1e9, "trackingSt')     # dernière ligne coupée
    (dst / "frames.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (dst / "manifest.json").write_text(json.dumps({
        "app": "SpatialCapture", "schemaVersion": 1, "state": "complete", "depthCaptureMode": "sceneDepth",
        "depth": "depth.f32", "confidence": "confidence.u8", "frames": "frames.jsonl",
        "depthFrameCount": len(rows), "videoWidth": 1920, "videoHeight": 1440}), encoding="utf-8")
    return dst, len(rows)


@pytest.fixture(scope="module")
def take(tmp_path_factory):
    base = tmp_path_factory.mktemp("sc")
    src, n_rows = make_take(base / "Take-test")
    out = base / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scan_to_domain.py"), str(src), "--out", str(out), "--dx", "0.1",
                        "--name", "piece", "--margin", "0"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    case = json.loads((out / "piece.json").read_text(encoding="utf-8"))
    return case, np.load(out / "solids.npy"), r.stdout, n_rows


def test_frame_selection(take):
    _, _, log, n_rows = take
    n_poses = n_rows // 3
    assert f"{n_rows - 1} images à suivi normal" in log, log
    # une image gardée par pose : les répétitions à 2 mm sont écartées
    kept = int(log.split("images à suivi normal, ")[1].split(" gardées")[0])
    assert kept == n_poses, (kept, n_poses)


def test_same_room(take):
    case, solids, _, _ = take
    g = case["grid"]
    dims = sorted([g["Lx"], g["Ly"]])
    assert abs(dims[0] - 3.0) <= 0.25 and abs(dims[1] - 4.0) <= 0.25, dims
    assert case["scan"]["ceiling_scanned"]
    assert abs(case["scan"]["ceiling_height"] - 2.5) < 0.06
    air = (solids == 0).sum() * 0.1 ** 3
    assert 25.0 < air < 31.0, air
    assert case["air"]["model"] == "anelastic"
