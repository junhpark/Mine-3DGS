#!/usr/bin/env python3
"""골든 게이트 — E57 핀홀 로컬 축 규약을 RGB 일치로 자동 판정한다.

점군에 RGB 가 있으므로 추측할 필요가 없다. 24 가지 축정렬 회전(부호 있는 순열,
det=+1)을 전부 시도해서, 점을 큐브 면에 투영한 뒤 그 점의 색과 JPEG 픽셀 색이
가장 잘 맞는 회전을 고른다. 맞는 규약은 색 오차가 뚜렷하게 낮다.

사용:
    python tools/resolve_pinhole_convention.py <file.e57>
    python tools/resolve_pinhole_convention.py <file.e57> --scan 60 --stride 40 --overlay out/gate
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
import pye57
from pye57 import libe57


# --------------------------------------------------------------------- E57 읽기
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


def load_faces(root, scan_guid):
    """해당 스캔에 속한 이미지들의 pose·내부파라미터·JPEG 를 읽는다."""
    faces = []
    vec = libe57.VectorNode(root.get("/images2D"))
    for i in range(vec.childCount()):
        s = libe57.StructureNode(vec.get(i))
        if _val(s, "associatedData3DGuid") != scan_guid:
            continue
        if not s.isDefined("pinholeRepresentation"):
            continue
        r = libe57.StructureNode(s.get("pinholeRepresentation"))
        W, H = int(_val(r, "imageWidth")), int(_val(r, "imageHeight"))
        f_m = _val(r, "focalLength")
        pw = _val(r, "pixelWidth")
        ph = _val(r, "pixelHeight")
        # pixelWidth 가 없으면 90도 화각(큐브맵)으로 가정
        fx = f_m / pw if (f_m and pw) else W / 2.0
        fy = f_m / ph if (f_m and ph) else H / 2.0
        cx = _val(r, "principalPointX", W / 2.0)
        cy = _val(r, "principalPointY", H / 2.0)

        pose = libe57.StructureNode(s.get("pose"))
        rot = libe57.StructureNode(pose.get("rotation"))
        tr = libe57.StructureNode(pose.get("translation"))
        q = [_val(rot, k, 0.0) for k in ("w", "x", "y", "z")]
        t = np.array([_val(tr, k, 0.0) for k in ("x", "y", "z")])

        blob = libe57.BlobNode(r.get("jpegImage"))
        buf = np.frombuffer(bytes(blob.read_buffer()), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR

        faces.append({
            "index": i, "name": _val(s, "name"),
            "K": (fx, fy, cx, cy), "W": W, "H": H,
            "R": quat_to_R(q), "t": t, "img": img,
        })
    return faces


# ------------------------------------------------------------ 후보 회전 24가지
def axis_rotations():
    """축정렬 회전 전부: 부호 있는 순열 중 det=+1 인 24개."""
    out = []
    for perm in itertools.permutations(range(3)):
        for sx, sy, sz in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for row, col in enumerate(perm):
                M[row, col] = (sx, sy, sz)[row]
            if abs(np.linalg.det(M) - 1.0) < 1e-9:
                out.append(M)
    return out


def label(M):
    names = ["X", "Y", "Z"]
    rows = []
    for r in range(3):
        c = int(np.argmax(np.abs(M[r])))
        rows.append(("-" if M[r, c] < 0 else "+") + names[c])
    return f"cam({rows[0]},{rows[1]},{rows[2]})"


# --------------------------------------------------------------------- 채점
def score(faces, pts, rgb, R_axis, collect=False):
    """투영된 점의 색과 이미지 픽셀 색의 평균 절대 오차. 낮을수록 맞다."""
    tot_err, tot_n, overlays = 0.0, 0, []
    for f in faces:
        fx, fy, cx, cy = f["K"]
        R_cw = R_axis @ f["R"].T
        t_cw = -R_cw @ f["t"]
        pc = pts @ R_cw.T + t_cw

        z = pc[:, 2]
        ok = z > 1e-6
        if not ok.any():
            continue
        u = fx * pc[ok, 0] / z[ok] + cx
        v = fy * pc[ok, 1] / z[ok] + cy
        inside = (u >= 0) & (u < f["W"]) & (v >= 0) & (v < f["H"])
        if inside.sum() < 100:
            continue

        ui = u[inside].astype(np.int32)
        vi = v[inside].astype(np.int32)
        px = f["img"][vi, ui][:, ::-1].astype(np.float32)       # BGR→RGB
        pc_rgb = rgb[ok][inside].astype(np.float32)

        tot_err += float(np.abs(px - pc_rgb).sum())
        tot_n += int(inside.sum()) * 3
        if collect:
            ov = f["img"].copy()
            step = max(1, len(ui) // 60000)
            ov[vi[::step], ui[::step]] = (0, 0, 255)
            overlays.append((f["name"], ov, int(inside.sum())))

    if tot_n == 0:
        return float("inf"), 0, overlays
    return tot_err / tot_n, tot_n // 3, overlays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e57")
    ap.add_argument("--scan", type=int, default=0, help="검증에 쓸 스캔 인덱스")
    ap.add_argument("--stride", type=int, default=50, help="점 서브샘플 간격")
    ap.add_argument("--overlay", help="최적 규약의 오버레이 저장 디렉토리")
    args = ap.parse_args()

    f57 = pye57.E57(args.e57)
    root = f57.image_file.root()

    scans = libe57.VectorNode(root.get("/data3D"))
    sn = libe57.StructureNode(scans.get(args.scan))
    guid = _val(sn, "guid")
    print(f"scan[{args.scan}] {_val(sn, 'name')}  guid={guid}")

    d = f57.read_scan(args.scan, colors=True, ignore_missing_fields=True)
    pts = np.column_stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]])[:: args.stride]
    rgb = np.column_stack([d["colorRed"], d["colorGreen"], d["colorBlue"]])[:: args.stride]
    print(f"점 {len(pts):,} 개 사용 (전체의 1/{args.stride})")
    print(f"점군 중심 {pts.mean(0).round(3)}  범위 {(pts.max(0) - pts.min(0)).round(2)} m")

    faces = load_faces(root, guid)
    print(f"면 {len(faces)} 장, 내부파라미터 fx={faces[0]['K'][0]:.1f} "
          f"(90도 화각이면 {faces[0]['W'] / 2:.0f})")
    print(f"카메라 중심 {faces[0]['t'].round(4)}\n")

    results = []
    for M in axis_rotations():
        err, n, _ = score(faces, pts, rgb, M)
        results.append((err, n, M))
    results.sort(key=lambda r: r[0])

    print(f"{'규약':<22} {'색오차':>8} {'투영점수':>10}")
    print("-" * 44)
    for err, n, M in results[:6]:
        print(f"{label(M):<22} {err:>8.2f} {n:>10,}")

    best_err, best_n, best_M = results[0]
    second = results[1][0]
    print(f"\n최적: {label(best_M)}")
    print(best_M.astype(int))
    print(f"색오차 {best_err:.2f} (채널당 0-255), 2위와 {second - best_err:.2f} 차이")
    if second - best_err < 5.0:
        print("!! 1-2위 차이가 작다. --scan 을 바꿔 다시 확인할 것")
    elif best_err > 40.0:
        print("!! 최적값도 오차가 크다. 축정렬 회전으로 설명되지 않는다")
    else:
        print("규약 확정 — manifest 에 기록하고 고정한다")

    if args.overlay:
        out = Path(args.overlay)
        out.mkdir(parents=True, exist_ok=True)
        _, _, ovs = score(faces, pts, rgb, best_M, collect=True)
        for name, ov, n in ovs:
            p = out / f"{str(name).replace(' ', '_')}.jpg"
            cv2.imwrite(str(p), cv2.resize(ov, (1024, 1024)))
            print(f"  {p}  ({n:,} 점)")

    f57.close()


if __name__ == "__main__":
    main()
