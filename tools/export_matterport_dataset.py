#!/usr/bin/env python3
"""Matterport E57 → 데이터셋 계약 (COLMAP sparse + 이미지 + 초기점 + 매니페스트).

규약은 tools/resolve_pinhole_convention.py 로 확정된 것을 고정해 쓴다:
    R_axis = diag(1, -1, -1)        E57 핀홀 로컬 → OpenCV 카메라
    R_cw   = R_axis @ R_img.T
    t_cw   = -R_cw @ t_img

좌표 프레임: 학습은 LOCAL_METRIC(선택 스윕 중심을 원점으로 평행이동, 1 unit = 1 m).
원 프레임으로 돌아가는 T_source_from_local 을 매니페스트에 기록한다.

사용:
    python tools/export_matterport_dataset.py <file.e57> data/bogo/dataset \
        --sweeps 4-120 --downscale 4 --point-stride 40 --voxel 0.05 --nadir-mask
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pye57
from pye57 import libe57

# 확정된 규약 — 바꾸지 말 것 (tools/resolve_pinhole_convention.py 로 검증)
R_AXIS = np.diag([1.0, -1.0, -1.0])
CONVENTION_ID = "e57_pinhole_camX_negY_negZ"


# ------------------------------------------------------------------ E57 유틸
def _val(node, path, default=None):
    try:
        if not node.isDefined(path):
            return default
        c = node.get(path)
        t = c.type()
        if t == libe57.E57_FLOAT:
            return libe57.FloatNode(c).value()
        if t == libe57.E57_INTEGER:
            return libe57.IntegerNode(c).value()
        if t == libe57.E57_SCALED_INTEGER:
            return libe57.ScaledIntegerNode(c).scaledValue()
        if t == libe57.E57_STRING:
            return libe57.StringNode(c).value()
    except Exception:
        pass
    return default


def quat_to_R(q):
    w, x, y, z = np.asarray(q, float)
    n = np.linalg.norm([w, x, y, z])
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def file_sha256(path: Path, skip=False):
    """스트리밍 SHA-256. 같은 파일은 사이드카에 캐시해 재계산하지 않는다."""
    if skip:
        return "0" * 64
    st = path.stat()
    cache = path.with_suffix(path.suffix + ".sha256")
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("size") == st.st_size and c.get("mtime") == int(st.st_mtime):
                return c["sha256"]
        except Exception:
            pass
    print(f"SHA-256 계산 중 ({st.st_size/1e9:.1f} GB, 한 번만)...", flush=True)
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
            done += len(chunk)
            print(f"\r  {done/st.st_size*100:5.1f}%", end="", flush=True)
    print()
    d = h.hexdigest()
    try:
        cache.write_text(json.dumps({"size": st.st_size, "mtime": int(st.st_mtime),
                                     "sha256": d}))
    except Exception:
        pass
    return d


def parse_range(spec, n):
    """'4-120' 또는 '4,7,10' 또는 'all'"""
    if spec in (None, "all"):
        return list(range(n))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [i for i in sorted(set(out)) if 0 <= i < n]


# ------------------------------------------------------------------ 메타 수집
def collect(root):
    scans = libe57.VectorNode(root.get("/data3D"))
    guid_to_idx, names = {}, {}
    for i in range(scans.childCount()):
        s = libe57.StructureNode(scans.get(i))
        guid_to_idx[_val(s, "guid")] = i
        names[i] = _val(s, "name")

    faces = defaultdict(list)
    vec = libe57.VectorNode(root.get("/images2D"))
    for i in range(vec.childCount()):
        s = libe57.StructureNode(vec.get(i))
        idx = guid_to_idx.get(_val(s, "associatedData3DGuid"))
        if idx is None or not s.isDefined("pinholeRepresentation"):
            continue
        r = libe57.StructureNode(s.get("pinholeRepresentation"))
        W, H = int(_val(r, "imageWidth")), int(_val(r, "imageHeight"))
        f_m, pw, ph = _val(r, "focalLength"), _val(r, "pixelWidth"), _val(r, "pixelHeight")
        pose = libe57.StructureNode(s.get("pose"))
        rot = libe57.StructureNode(pose.get("rotation"))
        tr = libe57.StructureNode(pose.get("translation"))
        faces[idx].append({
            "img_index": i,
            "name": _val(s, "name"),
            "W": W, "H": H,
            "fx": f_m / pw if (f_m and pw) else W / 2.0,
            "fy": f_m / ph if (f_m and ph) else H / 2.0,
            "cx": _val(r, "principalPointX", W / 2.0),
            "cy": _val(r, "principalPointY", H / 2.0),
            "R": quat_to_R([_val(rot, k, 0.0) for k in ("w", "x", "y", "z")]),
            "t": np.array([_val(tr, k, 0.0) for k in ("x", "y", "z")]),
            "node": r,
        })
    for v in faces.values():
        v.sort(key=lambda e: e["img_index"])
    return names, faces


# ------------------------------------------------------------------- PLY 쓰기
def write_ply(path, xyz, rgb, comment):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment {comment}\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def voxel_downsample(xyz, rgb, size):
    if size <= 0:
        return xyz, rgb
    key = np.floor(xyz / size).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    return xyz[first], rgb[first]


# --------------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e57")
    ap.add_argument("out")
    ap.add_argument("--sweeps", default="all", help="예: 4-120 / 4,7,10 / all")
    ap.add_argument("--downscale", type=int, default=4, help="이미지 축소 배율")
    ap.add_argument("--point-stride", type=int, default=40)
    ap.add_argument("--voxel", type=float, default=0.05, help="복셀 크기 m, 0 이면 생략")
    ap.add_argument("--max-points", type=int, default=3_000_000)
    ap.add_argument("--nadir-mask", action="store_true",
                    help="점 커버리지가 없는 나디르 영역 마스크 생성")
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--test-every", type=int, default=8,
                    help="N 스윕마다 하나를 테스트 그룹으로. 0 이면 테스트 없음")
    ap.add_argument("--init-groups", choices=["train", "all"], default="train",
                    help="초기점에 쓸 그룹. train 이면 테스트 스윕 점을 제외한다")
    ap.add_argument("--write-points3d", action="store_true",
                    help="points3D.txt 에도 초기점을 직접 기록 (스테이징이 안 채워줄 때)")
    ap.add_argument("--no-hash", action="store_true",
                    help="SHA-256 생략 (매니페스트 검증에 실패한다)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)

    f57 = pye57.E57(args.e57)
    root = f57.image_file.root()
    names, faces = collect(root)
    sel = parse_range(args.sweeps, len(names))
    sel = [i for i in sel if i in faces]
    print(f"스윕 {len(sel)} / {len(names)} 선택, 이미지 {sum(len(faces[i]) for i in sel)} 장")

    # LOCAL_METRIC 원점 = 선택 스윕 카메라 중심의 평균
    centers = np.array([faces[i][0]["t"] for i in sel])
    origin = centers.mean(axis=0)
    print(f"LOCAL_METRIC 원점 {origin.round(3)}  "
          f"카메라 분포 {(centers.max(0) - centers.min(0)).round(1)} m")

    # split 을 먼저 정한다 — 초기점에서 테스트 스윕을 실제로 빼야 하기 때문
    gid_of = {si: f"S{si:03d}" for si in sel}
    if args.test_every > 0:
        test_idx = {si for i, si in enumerate(sel) if i % args.test_every == args.test_every // 2}
    else:
        test_idx = set()
    train_gids = [gid_of[si] for si in sel if si not in test_idx]
    test_gids = [gid_of[si] for si in sel if si in test_idx]
    init_idx = [si for si in sel if args.init_groups == "all" or si not in test_idx]
    print(f"train {len(train_gids)} / test {len(test_gids)} 그룹, "
          f"초기점 스윕 {len(init_idx)} 개 ({args.init_groups})")

    ds = args.downscale
    f0 = faces[sel[0]][0]
    W, H = f0["W"] // ds, f0["H"] // ds
    fx, fy = f0["fx"] / ds, f0["fy"] / ds
    cx, cy = f0["cx"] / ds, f0["cy"] / ds
    print(f"이미지 {W}x{H}, fx={fx:.1f}, 화각 {2*np.degrees(np.arctan(W/2/fx)):.1f}도")

    # ---------------- cameras.txt / images.txt / 이미지 추출
    (out / "sparse" / "0" / "cameras.txt").write_text(
        "# camera_id, model, width, height, params[]\n"
        f"1 PINHOLE {W} {H} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n",
        encoding="utf-8")

    groups, lines, image_id = {}, [], 0
    nadir_acc = None
    for n, si in enumerate(sel, 1):
        gid = f"S{si:03d}"
        members = []
        for k, fc in enumerate(faces[si]):
            image_id += 1
            fname = f"{gid}_f{k}.jpg"
            blob = libe57.BlobNode(fc["node"].get("jpegImage"))
            img = cv2.imdecode(np.frombuffer(bytes(blob.read_buffer()), np.uint8),
                               cv2.IMREAD_COLOR)
            if ds != 1:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "images" / fname), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 94])

            R_cw = R_AXIS @ fc["R"].T
            t_cw = -R_cw @ (fc["t"] - origin)
            q = R_to_quat(R_cw)
            lines.append(f"{image_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                         f"{t_cw[0]:.10f} {t_cw[1]:.10f} {t_cw[2]:.10f} 1 {fname}")
            lines.append("")
            members.append(fname)
        groups[gid] = {"type": "tls_station", "members": members,
                       "source_translation": faces[si][0]["t"].round(4).tolist()}
        if n % 10 == 0 or n == len(sel):
            print(f"  이미지 {n}/{len(sel)} 스윕 처리")

    (out / "sparse" / "0" / "images.txt").write_text(
        "# image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name\n"
        "# (points2d line intentionally empty)\n" + "\n".join(lines) + "\n",
        encoding="utf-8")

    # ---------------- 점군 → init_points.ply (+ 나디르 커버리지)
    acc_xyz, acc_rgb = [], []
    for n, si in enumerate(init_idx, 1):
        d = f57.read_scan(si, colors=True, ignore_missing_fields=True)
        p = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]])
        c = np.column_stack([d["colorRed"], d["colorGreen"], d["colorBlue"]]).astype(np.uint8)

        if args.nadir_mask:
            fc = faces[si][5] if len(faces[si]) > 5 else None
            if fc is not None:
                R_cw = R_AXIS @ fc["R"].T
                t_cw = -R_cw @ fc["t"]
                pc = p[:: max(1, args.point_stride // 4)] @ R_cw.T + t_cw
                z = pc[:, 2]
                ok = z > 1e-6
                u = (fc["fx"] * pc[ok, 0] / z[ok] + fc["cx"]) / ds
                v = (fc["fy"] * pc[ok, 1] / z[ok] + fc["cy"]) / ds
                ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                cov = np.zeros((H, W), np.uint8)
                cov[v[ins].astype(np.int32), u[ins].astype(np.int32)] = 255
                nadir_acc = cov if nadir_acc is None else np.maximum(nadir_acc, cov)

        p = p[:: args.point_stride] - origin
        c = c[:: args.point_stride]
        if args.voxel > 0:
            p, c = voxel_downsample(p, c, args.voxel)
        acc_xyz.append(p.astype(np.float32))
        acc_rgb.append(c)
        if n % 10 == 0 or n == len(sel):
            print(f"  점군 {n}/{len(sel)} 스윕, 누적 {sum(len(a) for a in acc_xyz):,}")

    xyz = np.concatenate(acc_xyz)
    rgb = np.concatenate(acc_rgb)
    if args.voxel > 0:
        xyz, rgb = voxel_downsample(xyz, rgb, args.voxel)
    if len(xyz) > args.max_points:
        keep = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        keep.sort()
        xyz, rgb = xyz[keep], rgb[keep]
    write_ply(out / "init_points.ply", xyz, rgb, "frame=LOCAL_METRIC unit=m")
    print(f"init_points.ply: {len(xyz):,} 점, 범위 {(xyz.max(0)-xyz.min(0)).round(1)} m")

    (out / "sparse" / "0" / "points3D.txt").write_text(
        "# point3D_id, x, y, z, r, g, b, error, track[]\n"
        "# 초기점은 init_points.ply 에 있다 (스테이징이 여기로 변환)\n",
        encoding="utf-8")

    # ---------------- 나디르 마스크
    if args.nadir_mask and nadir_acc is not None:
        k = np.ones((9, 9), np.uint8)
        cov = cv2.morphologyEx(nadir_acc, cv2.MORPH_CLOSE, k, iterations=3)
        hole = (cov == 0).astype(np.uint8)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(hole, 8)
        mask = np.full((H, W), 255, np.uint8)   # 255 = 사용, 0 = 제외
        cid = lab[H // 2, W // 2]
        if cid != 0 and stats[cid, cv2.CC_STAT_AREA] > 0.01 * H * W:
            m = (lab == cid).astype(np.uint8) * 255
            m = cv2.dilate(m, k, iterations=2)
            mask[m > 0] = 0
            frac = (mask == 0).mean()
            (out / "masks").mkdir(exist_ok=True)
            cv2.imwrite(str(out / "masks" / "face5_nadir.png"), mask)
            print(f"나디르 마스크: 하면의 {frac*100:.1f}% 제외 → masks/face5_nadir.png")
        else:
            print("나디르 구멍을 찾지 못했다 — 마스크 생략")

    # ---------------- manifest
    e57_path = Path(args.e57)
    digest = file_sha256(e57_path, skip=args.no_hash)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or None
    except Exception:
        commit = None
    try:
        import minegs
        version = getattr(minegs, "__version__", "0.0.0")
    except Exception:
        version = "0.0.0"

    T = np.eye(4)
    T[:3, 3] = origin
    manifest = {
        "schema_version": "1.0",
        "dataset_id": args.dataset_id or f"{e57_path.stem}_s{sel[0]}-{sel[-1]}_d{ds}",
        "coordinate_frames": {
            "evaluation": "TLS_GLOBAL",
            "training": "LOCAL_METRIC",
            "unit": "m",
            "T_tls_from_local": T.tolist(),
        },
        "capture_groups": {g: {"type": v["type"], "members": v["members"]}
                           for g, v in groups.items()},
        "split": {
            "train_groups": train_gids,
            "test_groups": test_gids,
        },
        "initialization": {
            "source": "tls",
            "file": "init_points.ply",
            "groups": [gid_of[si] for si in init_idx],
        },
        "provenance": {
            "minegs_version": version,
            "git_commit": commit,
            "source_assets": [{"path": str(e57_path), "sha256": digest}],
        },
        "source": "tls",
        "scale": {"basis": "tls_pose", "factor": 1.0},
    }

    # 스키마가 금지하는 추가 기록은 사이드카로 분리한다
    (out / "export_notes.json").write_text(json.dumps({
        "pano_convention": {
            "id": CONVENTION_ID,
            "R_axis": R_AXIS.astype(int).tolist(),
            "verified_by": "tools/resolve_pinhole_convention.py",
            "verified_scans": [10, 60, 100],
        },
        "sweeps": {g: v["source_translation"] for g, v in groups.items()},
        "export": {"downscale": ds, "point_stride": args.point_stride,
                   "voxel_m": args.voxel, "init_point_count": int(len(xyz)),
                   "nadir_mask": bool(args.nadir_mask)},
        "frame_note": ("evaluation 은 스키마가 TLS_GLOBAL 리터럴만 허용해 그렇게 적었다. "
                       "실제로는 E57 파일 자체의 SOURCE 프레임이며 측지 좌표계가 아니다."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n완료: {out}")
    print(f"  images/   {image_id} 장  {W}x{H}")
    print(f"  test 그룹 {len(manifest['split']['test_groups'])} 개 (8스윕마다 1개)")
    print("  split 은 초안이다 — 형상 홀드아웃은 chainage 구간으로 다시 잡을 것")
    f57.close()


if __name__ == "__main__":
    main()
