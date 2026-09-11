#!/usr/bin/env python3
"""images2D 가 실제로 무엇인지 확인한다 (Phase 0B.2 설계용 프로브).

블롭은 --dump 를 줄 때만 읽는다. 그 외에는 메타데이터만 보므로 수 GB 파일에서도
몇 초면 끝난다.

사용:
    python probe_images2d.py <file.e57>
    python probe_images2d.py <file.e57> --dump out/probe   # 첫 스캔의 이미지 전부 저장
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pye57
from pye57 import libe57

REPRESENTATIONS = (
    "sphericalRepresentation",
    "pinholeRepresentation",
    "cylindricalRepresentation",
    "visualReferenceRepresentation",
)
BLOB_FIELDS = ("jpegImage", "pngImage")


def _val(node, path, default=None):
    try:
        if not node.isDefined(path):
            return default
        child = node.get(path)
        t = child.type()
        if t == libe57.E57_FLOAT:
            return libe57.FloatNode(child).value()
        if t == libe57.E57_INTEGER:
            return libe57.IntegerNode(child).value()
        if t == libe57.E57_SCALED_INTEGER:
            return libe57.ScaledIntegerNode(child).scaledValue()
        if t == libe57.E57_STRING:
            return libe57.StringNode(child).value()
    except Exception:
        pass
    return default


def _pose(node):
    if not node.isDefined("pose"):
        return None
    pose = libe57.StructureNode(node.get("pose"))
    q = t = None
    if pose.isDefined("rotation"):
        r = libe57.StructureNode(pose.get("rotation"))
        q = tuple(_val(r, k, 0.0) for k in ("w", "x", "y", "z"))
    if pose.isDefined("translation"):
        tr = libe57.StructureNode(pose.get("translation"))
        t = tuple(_val(tr, k, 0.0) for k in ("x", "y", "z"))
    return (q, t)


def scan_guid_map(root):
    """data3D guid -> (index, name)"""
    out = {}
    if not root.isDefined("/data3D"):
        return out
    vec = libe57.VectorNode(root.get("/data3D"))
    for i in range(vec.childCount()):
        s = libe57.StructureNode(vec.get(i))
        out[_val(s, "guid")] = (i, _val(s, "name"))
    return out


def read_images(root):
    entries = []
    if not root.isDefined("/images2D"):
        return entries
    vec = libe57.VectorNode(root.get("/images2D"))
    for i in range(vec.childCount()):
        s = libe57.StructureNode(vec.get(i))
        e = {
            "index": i,
            "name": _val(s, "name"),
            "guid": _val(s, "guid"),
            "assoc": _val(s, "associatedData3DGuid"),
            "pose": _pose(s),
            "rep": None,
            "node": s,
        }
        for rep in REPRESENTATIONS:
            if not s.isDefined(rep):
                continue
            r = libe57.StructureNode(s.get(rep))
            info = {
                "type": rep,
                "w": _val(r, "imageWidth"),
                "h": _val(r, "imageHeight"),
                "pixelWidth": _val(r, "pixelWidth"),
                "pixelHeight": _val(r, "pixelHeight"),
                "focalLength": _val(r, "focalLength"),
                "ppx": _val(r, "principalPointX"),
                "ppy": _val(r, "principalPointY"),
                "blob_field": None,
                "blob_bytes": 0,
                "rep_node": r,
            }
            for bf in BLOB_FIELDS:
                if r.isDefined(bf):
                    b = libe57.BlobNode(r.get(bf))
                    info["blob_field"] = bf
                    info["blob_bytes"] = int(b.byteCount())
                    break
            e["rep"] = info
            break
        entries.append(e)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e57")
    ap.add_argument("--dump", help="첫 스캔에 속한 이미지를 이 디렉토리에 저장")
    args = ap.parse_args()

    f = pye57.E57(args.e57)
    root = f.image_file.root()

    scans = scan_guid_map(root)
    imgs = read_images(root)
    print(f"data3D  : {len(scans)} 스캔")
    print(f"images2D: {len(imgs)} 장\n")

    kinds = Counter(e["rep"]["type"] if e["rep"] else "NONE" for e in imgs)
    sizes = Counter(
        (e["rep"]["w"], e["rep"]["h"]) if e["rep"] else None for e in imgs
    )
    print("표현 방식 :", dict(kinds))
    print("해상도    :", dict(sizes))

    linked = sum(1 for e in imgs if e["assoc"])
    matched = sum(1 for e in imgs if e["assoc"] in scans)
    print(f"\nassociatedData3DGuid 있음: {linked}/{len(imgs)}, data3D 와 일치: {matched}")

    by_scan = defaultdict(list)
    for e in imgs:
        by_scan[e["assoc"]].append(e)
    per = Counter(len(v) for v in by_scan.values())
    print(f"스캔당 이미지 수 분포: {dict(per)}")

    print("\n--- 첫 스캔에 속한 이미지 ---")
    first_guid = next((g for g in scans if g in by_scan), None)
    if first_guid is None:
        first_guid = next(iter(by_scan))
    idx, sname = scans.get(first_guid, ("?", "?"))
    print(f"scan[{idx}] {sname}  guid={first_guid}")
    group = by_scan[first_guid]
    for e in group:
        r = e["rep"] or {}
        q, t = e["pose"] if e["pose"] else (None, None)
        print(f"  [{e['index']:>4}] name={str(e['name'])[:28]:<28} {r.get('type')}")
        print(
            f"         {r.get('w')}x{r.get('h')}  "
            f"f={r.get('focalLength')}  pp=({r.get('ppx')}, {r.get('ppy')})  "
            f"{r.get('blob_bytes', 0)/1e6:.2f}MB"
        )
        if q:
            print(f"         q={tuple(round(v, 6) for v in q)}  t={tuple(round(v, 4) for v in t)}")

    # 같은 스캔 안의 이미지들이 광학중심을 공유하는지 (큐브맵이면 공유한다)
    ts = [e["pose"][1] for e in group if e["pose"] and e["pose"][1]]
    if len(ts) > 1:
        import itertools

        d = [
            max(abs(a[k] - b[k]) for k in range(3))
            for a, b in itertools.combinations(ts, 2)
        ]
        print(f"\n  그룹 내 translation 최대 차이: {max(d):.6f} m")
        print("  → 0 에 가까우면 큐브맵(같은 광학중심), 크면 별개 카메라 위치")

    if args.dump:
        out = Path(args.dump)
        out.mkdir(parents=True, exist_ok=True)
        for e in group:
            r = e["rep"]
            if not r or not r["blob_field"]:
                continue
            blob = libe57.BlobNode(r["rep_node"].get(r["blob_field"]))
            ext = "jpg" if r["blob_field"] == "jpegImage" else "png"
            p = out / f"img_{e['index']:04d}.{ext}"
            p.write_bytes(bytes(blob.read_buffer()))
            print(f"  저장: {p}")

    f.close()


if __name__ == "__main__":
    main()
